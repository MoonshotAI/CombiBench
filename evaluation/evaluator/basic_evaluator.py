import json
import os
import signal
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from threading import RLock, Event
from typing import TextIO
import time
import pathlib

from tqdm import tqdm
import tenacity
from datasets import Dataset, load_dataset
from pydantic import BaseModel, ConfigDict

from evaluation.client.lean_client import Lean4Client
from evaluation.client.llm_client import LLMClient
from evaluation.constant import (
    ErrorType,
    EvaluationResult,
    GenerationTask,
    LLMClientType,
    VerificationResult,
    VerificationTask,
)
from evaluation.verifier.one_stage_verify import one_stage_verify
from evaluation.util import logger


class BasicEvaluator(BaseModel):
    # dataset configuration
    dataset_name: str
    revision: str = "main"
    split: str
    index_column: str
    formal_column: str
    ground_truths_column: str | None = None
    response_column: str | None = None
    proc_response_hook: callable = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # lean4 verifier configuration
    lean_server_url: str
    lean_server_api_key: str | None = None
    lean_server_timeout: int = 60

    # llm configuration
    llm_client_type: LLMClientType = LLMClientType.OpenAI
    llm_server_url: str | None = None
    llm_server_api_key: str | None = None
    llm_name: str
    llm_server_timeout: float = 60.0

    # sampling parameters
    n: int
    temperature: float = 1.0
    max_tokens: int = 4096

    # prompt template
    system_prompt: str | None = None
    prompt_template: str

    # parallel settings
    n_generation_processes: int = 1
    n_verification_processes: int = 1

    # evaluation mode
    greedy_mode: bool = False
    max_turns: int = 1
    return_raw_text: bool = False
    check_formal_statements: bool = True

    # flags: should not be initialized by user
    model_config = ConfigDict(arbitrary_types_allowed=True)
    lock: RLock = RLock()
    successful_problems: set[str] = set()
    generation_done: int = 0
    verification_done: int = 0
    all_done: int = 0
    all_tasks: int = 0

    # interrupt handling
    _shutdown_event: Event = None
    _interrupted: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if pathlib.Path(self.prompt_template).exists():
            with open(self.prompt_template, "r") as f:
                self.prompt_template = f.read()

        self._shutdown_event = Event()
        self._interrupted = (
            False  # Flag to indicate if the evaluation has been interrupted
        )

    def _create_messages(
        self,
        formal_statement: str,
    ) -> list[dict[str, str]]:
        messages = []
        user_prompt = self.prompt_template.format(formal_statement=formal_statement)
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
        before_sleep=lambda state: logger.warning(
            f"LLM request retry attempt {state.attempt_number}/3 failed:"
            f" {type(state.outcome.exception()).__name__}: {str(state.outcome.exception())}..."
        ),
        retry=tenacity.retry_if_not_exception_type(KeyboardInterrupt),
    )
    def _send_request(
        self, llm_client: LLMClient, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        if self._check_interrupt():
            raise KeyboardInterrupt("Request cancelled due to shutdown")
        return llm_client.generate(messages)

    def _is_problem_successful(self, problem_id: str) -> bool:
        """Check if a problem has already been solved successfully."""
        return problem_id in self.successful_problems

    def _mark_problem_successful(self, problem_id: str):
        """Mark a problem as successfully solved."""
        self.successful_problems.add(problem_id)

    def load_hf_dataset(self) -> Dataset:
        return load_dataset(self.dataset_name, split=self.split, revision=self.revision)

    def get_initial_records(self) -> dict[str, EvaluationResult]:
        dataset = self.load_hf_dataset()
        records: dict[str, EvaluationResult] = {}
        for example in dataset:
            problem_id = example[self.index_column].strip()
            formal_statement = example[self.formal_column].strip()
            ground_truths = (
                example[self.ground_truths_column]
                if self.ground_truths_column
                else None
            )
            query_messages = self._create_messages(formal_statement)
            records[problem_id] = EvaluationResult(
                problem_id=problem_id,
                formal_statement=formal_statement,
                ground_truths=ground_truths,
                query_messages=query_messages,
                responses=[],
                solution_attempts=[],
                error_types=[],
                passed_at=[],
                correct_solutions=[],
                one_success_solution=None,
            )
        return records

    def save_config(self, output_dir: str):
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(
                self.model_dump(
                    exclude={
                        "proc_response_hook",
                        "lock",
                        "successful_problems",
                        "_shutdown_event",
                        "_interrupted",
                    }
                ),
                f,
                indent=4,
                ensure_ascii=False,
            )

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        if not self._interrupted:
            logger.info("Received interrupt signal, initiating graceful shutdown...")
            self._interrupted = True
            self._shutdown_event.set()

    def _check_interrupt(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event.is_set()

    def _create_generation_tasks(
        self,
        records: dict[str, EvaluationResult],
        generation_queue: Queue[GenerationTask],
    ) -> None:
        for record in records.values():
            for _ in range(self.n):
                task = GenerationTask(
                    problem_id=record.problem_id,
                    messages=record.query_messages,
                    formal_statement=record.formal_statement,
                    ground_truths=record.ground_truths,
                )
                generation_queue.put(task)
        logger.info(f"Created {len(records) * self.n} tasks")
        self.all_tasks = len(records) * self.n

    def _generator(
        self,
        llm_client: LLMClient,
        generation_queue: Queue[GenerationTask],
        judge_queue: Queue[VerificationTask],
        result_queue: Queue[VerificationResult],
    ):
        while not self._check_interrupt():
            try:
                task = generation_queue.get(timeout=0.1)

                with self.lock:
                    already_solved = self._is_problem_successful(task.problem_id)
                if already_solved and self.greedy_mode:
                    continue

                try:
                    model_output, finish_reason = self._send_request(
                        llm_client, task.messages
                    )
                except Exception:
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=ErrorType.GENERATION_TIMEOUT,
                            raw_text=None,
                            lean4_code=None,
                            answer_predicts=None,
                            lean_feedback=None,
                        )
                    )
                    continue

                if finish_reason is not None and finish_reason != "stop":
                    if finish_reason == "length":
                        error_type = ErrorType.GENERATION_TOO_LONG
                    elif finish_reason == "repeat":
                        error_type = ErrorType.GENERATION_REPEAT
                    else:
                        error_type = ErrorType.GENERATION_ERROR
                    result_queue.put(
                        VerificationResult(
                            problem_id=task.problem_id,
                            error_type=error_type,
                            raw_text=model_output,
                            lean4_code=None,
                            answer_predicts=None,
                            lean_feedback=None,
                        )
                    )
                    continue

                # get model output and put it into judge queue
                judge_queue.put(
                    VerificationTask(
                        problem_id=task.problem_id,
                        text=model_output,
                        formal_statement=task.formal_statement,
                        ground_truths=task.ground_truths,
                    )
                )

                with self.lock:
                    self.generation_done += 1

            except Empty:
                if self._check_interrupt():
                    break
                with self.lock:
                    if self.all_done == self.all_tasks:
                        break
                time.sleep(0.1)
                continue
            except Exception as e:
                if self._check_interrupt():
                    break
                logger.exception(e)
                continue

    def _judge(
        self,
        lean4_client: Lean4Client,
        _generation_queue: Queue[GenerationTask],  # reserved for overloading
        judge_queue: Queue[VerificationTask],
        result_queue: Queue[VerificationResult],
    ):
        """Process verification tasks and put results in result queue."""
        while not self._check_interrupt():
            try:
                task = judge_queue.get(timeout=0.1)

                # Check if this problem is already solved in greedy mode
                with self.lock:
                    already_solved = self._is_problem_successful(task.problem_id)
                if already_solved and self.greedy_mode:
                    continue

                error_type, lean_feedback, lean4_code, answers = one_stage_verify(
                    text=task.text,
                    formal_statement=task.formal_statement,
                    lean4_client=lean4_client,
                    ground_truths=task.ground_truths,
                    return_raw_text=self.return_raw_text,
                    check_formal_statements=self.check_formal_statements,
                )
                result_queue.put(
                    VerificationResult(
                        problem_id=task.problem_id,
                        error_type=error_type,
                        raw_text=task.text,
                        lean4_code=lean4_code,
                        answer_predicts=answers,
                        lean_feedback=lean_feedback,
                    )
                )
            except Empty:
                if self._check_interrupt():
                    break
                with self.lock:
                    if self.all_done == self.all_tasks:
                        break
                time.sleep(0.1)
                continue
            except Exception as e:
                if self._check_interrupt():
                    break
                logger.exception(e)
                continue

    def _recorder(
        self,
        result_queue: Queue[VerificationResult],
        records: dict[str, EvaluationResult],
        output_file: TextIO,
    ):
        """Record results from verification queue to records and output file."""
        pbar = tqdm(total=self.all_tasks, desc="Processing", unit="task")
        n_success = 0
        estimated_wins = 0

        while not self._check_interrupt():
            try:
                result = result_queue.get(timeout=0.1)
                with self.lock:
                    record = records[result.problem_id]
                    record.responses.append(result.raw_text)
                    record.solution_attempts.append(result.lean4_code)
                    record.error_types.append(result.error_type)
                    if result.error_type == ErrorType.SUCCESS:
                        record.correct_solutions.append(result.lean4_code)
                        record.one_success_solution = result.lean4_code
                        record.passed_at.append(len(record.solution_attempts))
                        # Mark this problem as successful
                        if not self._is_problem_successful(result.problem_id):
                            self._mark_problem_successful(result.problem_id)
                            n_success += 1

                            # In greedy mode, skip remaining attempts for this problem
                            # and update the estimated wins
                            if self.greedy_mode:
                                winrate_current_problem = 1 / len(
                                    record.solution_attempts
                                )
                                remaining_attempts = self.n - len(
                                    record.solution_attempts
                                )
                                est_wins = int(
                                    winrate_current_problem * remaining_attempts
                                )
                                estimated_wins += est_wins + 1
                                self.all_done += remaining_attempts
                                logger.info(
                                    f"{result.problem_id} solved at attempt {len(record.solution_attempts)},"
                                    f" estimated {est_wins} wins in remaining {remaining_attempts} attempts"
                                )

                    self.all_done += 1
                    passrate = len(self.successful_problems) / len(records) * 100
                    accuracy = (
                        n_success / self.all_done * 100 if self.all_done > 0 else 0
                    )

                # Update progress bar
                if self.greedy_mode:
                    postfix_dict = {
                        "accuracy": f"{estimated_wins / self.all_done * 100:.2f}% est. ({estimated_wins}/{self.all_done})"
                    }
                else:
                    postfix_dict = {
                        "accuracy": f"{accuracy:.2f}% ({n_success}/{self.all_done})"
                    }

                if len(records) > 0:
                    postfix_dict["passrate"] = (
                        f"{passrate:.2f}% ({len(self.successful_problems)}/{len(records)})"
                    )

                pbar.set_postfix(postfix_dict)
                pbar.update(self.all_done - pbar.n)

                # Write real-time response
                self._write_real_time_response(result, record, output_file)

            except Empty:
                if self._check_interrupt():
                    break
                # Check if all tasks are actually done
                with self.lock:
                    if self.all_done == self.all_tasks:
                        break
                time.sleep(0.05)
                continue
            except Exception as e:
                if self._check_interrupt():
                    break
                logger.exception(f"Recorder error: {e}")
                continue

        pbar.close()

    def _write_real_time_response(
        self, result: VerificationResult, record: EvaluationResult, output_file: TextIO
    ):
        """Write real-time response to output file."""
        real_time_response = {
            "problem_id": result.problem_id,
            "formal_statement": record.formal_statement,
            "ground_truths": record.ground_truths,
            "query_messages": record.query_messages,
            "raw_text": result.raw_text,
            "lean4_code": result.lean4_code,
            "error_type": result.error_type,
            "lean_feedback": result.lean_feedback,
        }
        if not output_file.closed:
            output_file.write(json.dumps(real_time_response, ensure_ascii=False) + "\n")
            output_file.flush()

    def run_evaluation(
        self,
        llm_client: LLMClient,
        lean4_client: Lean4Client,
        output_file: TextIO,
    ) -> dict[str, EvaluationResult]:
        # Set up signal handlers for graceful interrupt
        signal.signal(signal.SIGINT, self._signal_handler)

        generation_queue = Queue()
        judge_queue = Queue()
        result_queue = Queue()

        # Get initial records using the standard method
        records = self.get_initial_records()

        assert self.all_done == 0, "all_done must be 0 before running evaluation"
        self._create_generation_tasks(records, generation_queue)

        executor = ThreadPoolExecutor(
            max_workers=self.n_generation_processes + self.n_verification_processes + 1,
            thread_name_prefix="easy_verify",
        )

        generator_futures = [
            executor.submit(
                self._generator,
                llm_client,
                generation_queue,
                judge_queue,
                result_queue,
            )
            for _ in range(self.n_generation_processes)
        ]

        judge_futures = [
            executor.submit(
                self._judge,
                lean4_client,
                generation_queue,
                judge_queue,
                result_queue,
            )
            for _ in range(self.n_verification_processes)
        ]

        recorder_future = executor.submit(
            self._recorder, result_queue, records, output_file
        )

        try:
            # Wait for all futures to complete with interrupt checking
            all_futures = generator_futures + judge_futures + [recorder_future]
            logger.info(f"Starting evaluation with {len(all_futures)} total threads")

            while all_futures and not self._check_interrupt():
                done, not_done = concurrent.futures.wait(all_futures, timeout=0.1)
                all_futures = list(not_done)

                with self.lock:
                    if self.all_done == self.all_tasks:
                        logger.info(
                            f"All tasks completed: {self.all_done}/{self.all_tasks}"
                        )
                        break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            self._signal_handler(signal.SIGINT, None)

        except Exception as e:
            if self._check_interrupt():
                logger.info("Interrupted by exception, shutting down...")
            else:
                logger.exception(e)

        finally:
            executor.shutdown(wait=False)
            logger.info("Done Evaluation.")
            return records
