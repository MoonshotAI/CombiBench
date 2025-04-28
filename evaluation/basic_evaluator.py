import json
import os
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import logging
import tenacity
from datasets import Dataset, load_dataset
from pydantic import BaseModel

from evaluation.constant import (
    LLMClientType,
    EvaluationResult,
    GenerationTask,
)
from evaluation.client.lean_client import Lean4Client
from evaluation.client.llm_client import LLMClient
from evaluation.util import logger


class BasicEvaluator(BaseModel):
    # dataset configuration
    dataset_name: str
    split: str
    index_column: str
    formal_column: str
    ground_truths_column: str | None = None

    # lean4 verifier configuration
    lean_server_url: str
    lean_server_api_key: str | None = None

    # llm configuration
    llm_client_type: LLMClientType = LLMClientType.OpenAI
    llm_server_url: str | None = None
    llm_server_api_key: str | None = None
    llm_name: str

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

    # flags
    generation_done: int = 0
    verification_done: int = 0
    all_done: int = 0
    all_tasks: int = 0

    def _create_messages(
        self, formal_statement: str,
    ) -> list[dict[str, str]]:
        messages = []
        user_prompt = self.prompt_template.format(
            formal_statement=formal_statement
        )
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        return messages

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=15),
        before_sleep=tenacity.before_sleep_log(logger, logging.ERROR),
    )
    def _send_request(
        self, llm_client: LLMClient, messages: list[dict[str, str]]
    ) -> tuple[str, str]:
        return llm_client.generate(messages)

    def load_hf_dataset(self) -> Dataset:
        return load_dataset(self.dataset_name, split=self.split)

    def get_initial_records(self) -> dict[str, EvaluationResult]:
        dataset = self.load_hf_dataset()
        records: dict[str, EvaluationResult] = {}
        for example in dataset:
            problem_id = example[self.index_column].strip()
            formal_statement = example[self.formal_column].strip()
            ground_truths = example[self.ground_truths_column] if self.ground_truths_column else None
            records[problem_id] = EvaluationResult(
                problem_id=problem_id,
                formal_statement=formal_statement,
                ground_truths=ground_truths,
                solution_attempts=[],
                texts=[],
                error_types=[],
                passed_at=[],
                correct_solutions=[],
                one_success_solution=None,
            )
        return records

    def save_config(self, output_dir: str):
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(self.model_dump(), f, indent=4, ensure_ascii=False)

    def _create_generation_tasks(
        self,
        records: dict[str, EvaluationResult],
        generation_queue: Queue[GenerationTask],
    ) -> None:
        for record in records.values():
            for _ in range(self.n):
                task = GenerationTask(
                    problem_id=record.problem_id,
                    messages=self._create_messages(record.formal_statement),
                    formal_statement=record.formal_statement,
                    ground_truths=record.ground_truths,
                )
                generation_queue.put(task)
        logger.info(f'Created {len(records) * self.n} tasks')
        self.all_tasks = len(records) * self.n

    def _generator(self):
        raise NotImplementedError

    def _judge(self):
        raise NotImplementedError

    def _recorder(self):
        raise NotImplementedError

    def run_evaluation(
        self,
        records: dict[str, EvaluationResult],
        llm_client: LLMClient,
        lean4_client: Lean4Client,
    ):
        generation_queue = Queue()
        judge_queue = Queue()
        result_queue = Queue()

        assert self.all_done == 0, 'all_done must be 0 before running evaluation'
        self._create_generation_tasks(records, generation_queue)

        try:
            # Create and start all workers
            with ThreadPoolExecutor(
                max_workers=self.n_generation_processes
                + self.n_verification_processes
                + 1
            ) as executor:
                # Start generator workers
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

                # Start judge workers
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

                # Start recorder worker
                recorder_future = executor.submit(self._recorder, result_queue, records)

                # Wait for all tasks to complete
                for future in generator_futures + judge_futures + [recorder_future]:
                    future.result()

        except Exception as e:
            logger.exception(e)
        finally:
            return records
