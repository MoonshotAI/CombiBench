import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import click
import pyjson5


from evaluation.constant import (
    SIMP_TACTIC,
    NEGATION_TACTIC,
    ANSWER_TACTIC,
    ErrorType,
    EvaluationResult,
    ProofStage,
    GenerationTask,
    VerificationResult,
    VerificationTask,
)
from evaluation.client.lean_client import Lean4Client, verify
from evaluation.client.llm_client import LLMClient
from evaluation.basic_evaluator import BasicEvaluator
from evaluation.util import logger, extract_code_and_answer_from_text


class OnlineTwoStageEvaluator(BasicEvaluator):

    def _generator(
        self,
        llm_client: LLMClient,
        generation_queue: Queue[GenerationTask],
        judge_queue: Queue[VerificationTask],
        result_queue: Queue[VerificationResult],
    ):
        while True:
            try:
                task = generation_queue.get(timeout=1)
                model_output, finish_reason = self._send_request(llm_client, task.messages)

                if finish_reason != 'stop':
                    logger.debug(f'{finish_reason=}, check your generation setting and llm client!')
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=ErrorType.GENERATION_ERROR,
                            raw_text=model_output,
                            lean4_code=None,
                            answer_predicts=None,
                            lean_feedback=None,
                        )
                    )
                    continue

                code, _ = extract_code_and_answer_from_text(
                    text=model_output,
                    formal_statements=task.formal_statement
                )

                if code is None:
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=ErrorType.FORMAT_ERROR,
                            raw_text=model_output,
                            lean4_code=None,
                            answer_predicts=None,
                            lean_feedback=None,
                        )
                    )
                else:
                    judge_task = VerificationTask(
                        problem_id=task.problem_id,
                        text=model_output,
                        stage=task.stage,
                        formal_statement=task.formal_statement,
                        ground_truths=task.ground_truths,
                    )
                    judge_queue.put(judge_task)

                self.generation_done += 1

            except Empty:
                if self.all_done == self.all_tasks:
                    break
                time.sleep(0.5)
                continue
            except Exception as e:
                logger.exception(e)
                continue

    def _judge(
        self,
        lean_client: Lean4Client,
        generation_queue: Queue[GenerationTask],
        judge_queue: Queue[VerificationTask],
        result_queue: Queue[VerificationResult],
    ):
        while True:
            try:
                task = judge_queue.get(timeout=1)
                lean4_code, answers = extract_code_and_answer_from_text(
                    text=task.text,
                    formal_statements=task.formal_statement,
                )
                if task.stage == ProofStage.SIMP:
                    for tag, answer in zip(answers.keys(), task.ground_truths):
                        lean4_code += '\n' + SIMP_TACTIC.format(
                            answer_tag=tag, answer=answer
                        )
                elif task.stage == ProofStage.NEGATION:
                    for tag, answer in zip(answers.keys(), task.ground_truths):
                        lean4_code += '\n' + NEGATION_TACTIC.format(
                            answer_tag=tag, answer=answer
                        )

                is_valid, lean_feedback = verify(
                    lean4_code,
                    lean_client,
                )

                if task.stage == ProofStage.PROOF:  # stage 1: generate answer and proof
                    if not is_valid:
                        result_queue.put(
                            VerificationResult(
                                problem_id=task.problem_id,
                                error_type=ErrorType.PROOF_FAILED,
                                raw_text=task.text,
                                lean4_code=lean4_code,
                                answer_predicts=answers,
                                lean_feedback=lean_feedback,
                            )
                        )
                    else:
                        if task.ground_truths is None:
                            result_queue.put(
                                VerificationResult(
                                    problem_id=task.problem_id,
                                    error_type=ErrorType.SUCCESS,
                                    raw_text=task.text,
                                    lean4_code=lean4_code,
                                    answer_predicts=answers,
                                    lean_feedback=lean_feedback,
                                )
                            )
                        elif answers is None:
                            result_queue.put(
                                VerificationResult(
                                    problem_id=task.problem_id,
                                    error_type=ErrorType.ANSWER_NOT_MATCHED,
                                    raw_text=task.text,
                                    lean4_code=lean4_code,
                                    answer_predicts=answers,
                                    lean_feedback=lean_feedback,
                                )
                            )
                        elif (len(answers) == len(task.ground_truths)
                            and all(
                                answer == answer_predict
                                for answer, answer_predict in zip(
                                    task.ground_truths, answers.values()
                                )
                            )
                        ):  # answer is exact match, put it in the result queue
                            result_queue.put(
                                VerificationResult(
                                    problem_id=task.problem_id,
                                    error_type=ErrorType.SUCCESS,
                                    raw_text=task.text,
                                    lean4_code=lean4_code,
                                    answer_predicts=answers,
                                    lean_feedback=lean_feedback,
                                )
                            )
                        else:  # answer is not exact match, try some simple tactics to prove that the answer is correct
                            judge_queue.put(
                                VerificationTask(
                                    problem_id=task.problem_id,
                                    text=task.text,
                                    stage=ProofStage.SIMP,
                                    formal_statement=task.formal_statement,
                                    ground_truths=task.ground_truths,
                                )
                            )

                elif task.stage == ProofStage.SIMP:  # stage 2: try simple tactics
                    if is_valid:  # answers are proven correct, put it in the result queue
                        result_queue.put(
                            VerificationResult(
                                    problem_id=task.problem_id,
                                    error_type=ErrorType.SUCCESS,
                                    raw_text=task.text,
                                    lean4_code=lean4_code,
                                    answer_predicts=answers,
                                    lean_feedback=lean_feedback,
                            )
                        )
                    else:  # try to prove that the answer is incorrect
                        judge_queue.put(
                            VerificationTask(
                                problem_id=task.problem_id,
                                text=task.text,
                                stage=ProofStage.NEGATION,
                                formal_statement=task.formal_statement,
                                ground_truths=task.ground_truths,
                            )
                        )

                elif task.stage == ProofStage.NEGATION:  # stage 3: try generation
                    # NOTE: if we verified a proof by negation, that must be something wrong
                    # with the statement
                    if is_valid:  # negation is proven, put it in the result queue
                        result_queue.put(
                            VerificationResult(
                                problem_id=task.problem_id,
                                error_type=ErrorType.ANSWER_PROOF_FAILED,
                                raw_text=task.text,
                                stage=ProofStage.NEGATION,
                                lean4_code=lean4_code,
                                lean_feedback=lean_feedback,
                                answer_predicts=answers,
                            )
                        )
                    else:  # negation is invalid, try generation
                        for tag, answer in zip(
                            answers.keys(), task.ground_truths
                        ):
                            new_formal_statement = ANSWER_TACTIC.format(
                                answer_tag=tag, answer=answer
                            )
                            lean4_code += '\n' + new_formal_statement
                        new_messages = self._create_messages(formal_statement=lean4_code,)
                        generation_queue.put(
                            GenerationTask(
                                problem_id=task.problem_id,
                                messages=new_messages,
                                formal_statement=lean4_code,
                                stage=ProofStage.ANSWER,
                                ground_truths=task.ground_truths,
                            )
                        )

                elif task.stage == ProofStage.ANSWER:  # stage 4: finally
                    if is_valid:  # answer is proven correct, put it in the result queue
                        error_type = ErrorType.SUCCESS
                    else:
                        error_type = ErrorType.ANSWER_PROOF_FAILED
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=error_type,
                            raw_text=task.text,
                            stage=ProofStage.ANSWER,
                            lean4_code=lean4_code,
                            answer_predicts=answers,
                            lean_feedback=lean_feedback
                        )
                    )
                else:
                    logger.error(f'unknown stage: {task.stage}')
                    exit()

                self.verification_done += 1

            except Empty:
                if self.all_done == self.all_tasks:
                    break
                time.sleep(0.5)
                continue

            except Exception as e:
                logger.exception(e)
                continue

    def _recorder(
        self,
        result_queue: Queue[VerificationResult],
        records: dict[str, EvaluationResult],
    ):
        while True:
            try:
                result = result_queue.get(timeout=1)
                record = records[result.problem_id]
                record.texts.append(result.raw_text)
                record.solution_attempts.append(result.lean4_code)
                record.error_types.append(result.error_type)
                if result.error_type == ErrorType.SUCCESS:
                    record.correct_solutions.append(result.lean4_code)
                    record.n_correct_proofs += 1
                    record.one_success_solution = result.lean4_code
                    # Store the current attempt index (0-based) when success occurs
                    record.passed_at.append(len(record.solution_attempts))

                self.all_done += 1
                logger.debug(
                    f'Processed {self.all_done}/{self.all_tasks} tasks, {self.all_done / self.all_tasks * 100:.2f}%'
                )

            except Empty:
                if self.all_done == self.all_tasks:
                    break
                time.sleep(0.5)
                continue
            except Exception as e:
                logger.exception(e)
                continue


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True)
@click.option('--output-dir', type=click.Path(), default='./eval_logs')
def main(
    config: str,
    output_dir: str,
):
    config = pyjson5.loads(open(config).read())
    evaluator = OnlineTwoStageEvaluator(**config)

    model_name = evaluator.llm_name.split('/')[-1]
    output_dir = (
        Path(output_dir)
        / f'{model_name}-pass@{evaluator.n}-t{evaluator.temperature}-tks{evaluator.max_tokens}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.save_config(output_dir)

    logger.add(Path(output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info(f'Output directory: {output_dir}')

    records = evaluator.get_initial_records()
    logger.info(
        f'Loaded {len(records)} records from {evaluator.dataset_name} {evaluator.split} split'
    )
    logger.info(f'One example:\n{records[list(records.keys())[0]]}')

    llm_client = LLMClient(
        llm_client_type=evaluator.llm_client_type,
        llm_name=evaluator.llm_name,
        llm_server_url=evaluator.llm_server_url,
        llm_server_api_key=evaluator.llm_server_api_key,
        system_prompt=evaluator.system_prompt,
        temperature=evaluator.temperature,
        max_tokens=evaluator.max_tokens,
    )

    lean4_client = Lean4Client(
        base_url=evaluator.lean_server_url,
        api_key=evaluator.lean_server_api_key,
    )

    # Run evaluation
    results = evaluator.run_evaluation(records, llm_client, lean4_client)

    output_file = (
        output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    with open(output_file, 'w') as f:
        for result in results.values():
            f.write(json.dumps(asdict(result), ensure_ascii=False) + '\n')

    logger.info(f'Results saved to {output_file}')

    # Save pass@n, error types and one success code each if any
    statistics_file = output_dir / 'statistics.json'

    error_types = {}
    for record in records.values():
        for err_type in record.error_types:
            error_types[err_type.name] = error_types.get(err_type.name, 0) + 1
    logger.info(f'Error Types: {error_types}')

    pass_at_ns = {}
    i = 0

    while 2**i <= evaluator.n:
        n = 2**i
        is_pass = []
        for record in records.values():
            if not record.error_types:
                is_pass.append(False)
                continue

            # Take first n error types and check if any is SUCCESS
            is_pass.append(
                any(
                    err_type == ErrorType.SUCCESS for err_type in record.error_types[:n]
                )
            )
        pass_at_n = sum(is_pass) / len(is_pass)
        pass_at_ns[f'pass@{n}'] = pass_at_n
        logger.info(f'Pass@{n}: {pass_at_n * 100:.2f}%')
        i += 1

    one_success_codes = {}
    for record in records.values():
        if record.one_success_solution is not None:
            one_success_codes[record.problem_id] = record.one_success_solution

    with open(statistics_file, 'w') as f:
        json.dump(
            {
                'pass_at_n': pass_at_ns,
                'error_types': error_types,
                'success_cases': one_success_codes,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logger.info(f'statistics saved to {statistics_file}')


if __name__ == '__main__':
    main()
