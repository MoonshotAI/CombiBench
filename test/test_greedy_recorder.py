import io
from queue import Queue
from threading import Thread

import pytest

from evaluation.constant import ErrorType, EvaluationResult, VerificationResult
from evaluation.evaluator.basic_evaluator import BasicEvaluator


def make_evaluator(greedy):
    return BasicEvaluator(
        dataset_name="fixture",
        split="test",
        index_column="problem_id",
        formal_column="formal_statement",
        lean_server_url="http://unused.invalid",
        llm_name="unused",
        n=2,
        prompt_template="{formal_statement}",
        greedy_mode=greedy,
    )


def make_record(problem_id):
    return EvaluationResult(
        problem_id=problem_id,
        formal_statement="fixture",
        responses=[],
        solution_attempts=[],
        error_types=[],
        passed_at=[],
        correct_solutions=[],
    )


def record_results(evaluator, records, results):
    queue = Queue()
    for result in results:
        queue.put(result)
    evaluator.all_tasks = len(records) * evaluator.n
    output = io.StringIO()
    recorder = Thread(target=evaluator._recorder, args=(queue, records, output))
    recorder.start()
    recorder.join(timeout=2)
    completed = not recorder.is_alive()
    if not completed:
        evaluator._shutdown_event.set()
        recorder.join(timeout=2)
    assert not recorder.is_alive(), "recorder did not shut down during test cleanup"
    return completed, output.getvalue().splitlines()


@pytest.mark.parametrize("late_error", [ErrorType.SUCCESS, ErrorType.PROOF_FAILED])
def test_greedy_recorder_does_not_count_inflight_results_twice(late_error):
    evaluator = make_evaluator(greedy=True)
    records = {name: make_record(name) for name in ("first", "second")}
    completed, lines = record_results(
        evaluator,
        records,
        [
            VerificationResult("first", ErrorType.SUCCESS, "first success"),
            VerificationResult("first", late_error, "late completed attempt"),
            VerificationResult("second", ErrorType.PROOF_FAILED, "second failure"),
            VerificationResult("second", ErrorType.SUCCESS, "second success"),
        ],
    )

    assert evaluator.all_done == evaluator.all_tasks == 4
    assert completed, "greedy accounting left the recorder waiting for nonexistent work"
    assert records["first"].responses == ["first success"]
    assert records["second"].responses == ["second failure", "second success"]
    assert len(lines) == 3


def test_non_greedy_recorder_keeps_results_after_first_success():
    evaluator = make_evaluator(greedy=False)
    records = {"first": make_record("first")}
    completed, lines = record_results(
        evaluator,
        records,
        [
            VerificationResult("first", ErrorType.SUCCESS, "first success"),
            VerificationResult("first", ErrorType.PROOF_FAILED, "later failure"),
        ],
    )

    assert completed
    assert evaluator.all_done == evaluator.all_tasks == 2
    assert records["first"].responses == ["first success", "later failure"]
    assert len(lines) == 2
