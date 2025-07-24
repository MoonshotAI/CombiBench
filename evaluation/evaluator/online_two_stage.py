import time
from queue import Empty, Queue

from ..constant import (
    ErrorType,
    ProofStage,
    SIMP_TACTIC,
    NEGATION_TACTIC,
    ANSWER_TACTIC,
    VerificationTask,
    VerificationResult,
    GenerationTask,
)
from .basic_evaluator import BasicEvaluator
from ..util import extract_code_and_answer_from_text, logger
from ..client.lean_client import verify, Lean4Client
from ..client.llm_client import LLMClient


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
                try:
                    model_output, finish_reason = self._send_request(
                        llm_client, task.messages
                    )
                except Exception:
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=ErrorType.GENERATION_ERROR,
                            raw_text=None,
                            lean4_code=None,
                            answer_predicts=None,
                            lean_feedback=None,
                        )
                    )
                    continue

                if finish_reason is not None and finish_reason != "stop":
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
                    text=model_output, formal_statements=task.formal_statement
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
                        lean4_code += "\n" + SIMP_TACTIC.format(
                            answer_tag=tag, answer=answer
                        )
                elif task.stage == ProofStage.NEGATION:
                    for tag, answer in zip(answers.keys(), task.ground_truths):
                        lean4_code += "\n" + NEGATION_TACTIC.format(
                            answer_tag=tag, answer=answer
                        )

                is_valid, lean_feedback = verify(
                    lean4_code,
                    lean_client,
                    timeout=self.lean_server_timeout,
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
                        elif len(answers) == len(task.ground_truths) and all(
                            answer == answer_predict
                            for answer, answer_predict in zip(
                                task.ground_truths, answers.values()
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
                    if (
                        is_valid
                    ):  # answers are proven correct, put it in the result queue
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
                        for tag, answer in zip(answers.keys(), task.ground_truths):
                            new_formal_statement = ANSWER_TACTIC.format(
                                answer_tag=tag, answer=answer
                            )
                            lean4_code += "\n" + new_formal_statement
                        new_messages = self._create_messages(
                            formal_statement=lean4_code,
                        )
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
                            lean_feedback=lean_feedback,
                        )
                    )
                else:
                    logger.error(f"unknown stage: {task.stage}")
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
