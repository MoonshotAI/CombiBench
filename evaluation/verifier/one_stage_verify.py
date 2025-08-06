from typing import Any

from evaluation.client.lean_client import Lean4Client, verify
from evaluation.constant import ErrorType
from evaluation.util import extract_code_and_answer_from_text


def one_stage_verify(
    text: str,
    formal_statement: str | list[str],
    lean4_client: Lean4Client,
    ground_truths: list[str] | None = None,
    return_raw_text: bool = False,
    check_formal_statements: bool = True,
) -> tuple[ErrorType, dict[str, Any], str, dict[str, str]]:
    """verify a proof and its answers in a single stage.

    This function takes a text containing a proof and answers, verifies the proof's validity,
    and checks if the answers match the ground truths (if provided).

    Args:
        text (str): The input text containing the proof and answers.
        formal_statement (str | list[str]): The formal statement(s) to be proved. If a string is provided,
            it will be split by double newlines into a list of statements.
        lean4_client (Lean4Client): The Lean4 client instance used for verification.
        ground_truths (list[str] | None, optional): List of ground truth answers to compare against.
            If provided and no answers are found in the text, returns ANSWER_NOT_MATCHED error.

    Returns:
        tuple[ErrorType, dict[str, Any], str, list[str]]: A tuple containing:
            - ErrorType: The result of the evaluation (SUCCESS, FORMAT_ERROR, PROOF_FAILED, etc.)
            - dict[str, Any]: Feedback from the Lean4 verification
            - str: The extracted Lean4 code
            - dict[str, str]: The extracted answers
    """

    lean4_code, answers = extract_code_and_answer_from_text(
        text=text,
        formal_statements=formal_statement,
        allow_no_header=True,
        return_raw_text=return_raw_text,
        check_formal_statements=check_formal_statements,
    )
    lean_feedback = {}

    if lean4_code is None:
        return ErrorType.FORMAT_ERROR, lean_feedback, lean4_code, answers

    if ground_truths is not None and answers is None:
        return ErrorType.ANSWER_NOT_MATCHED, lean_feedback, lean4_code, answers

    if answers is not None and ground_truths is not None:
        for tag, ground_truth in zip(answers.keys(), ground_truths):
            lean4_code += f"\n\nexample: {tag} = {ground_truth} := by\n  try rfl\n  try norm_num"
    is_answers_valid, lean_feedback = verify(lean4_code, lean4_client)
    if is_answers_valid:
        return ErrorType.SUCCESS, lean_feedback, lean4_code, answers
    else:
        # return ErrorType.WRONG_ANSWERS, lean_feedback, lean4_code, answers
        return ErrorType.PROOF_FAILED, lean_feedback, lean4_code, answers
