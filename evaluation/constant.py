from dataclasses import dataclass
from uuid import UUID

from strenum import StrEnum

LEAN4_DEFAULT_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat"""

SIMP_TACTIC = """
example: {answer_tag} = {answer} := by
  try rfl
  try norm_num"""

NEGATION_TACTIC = """
example: {answer_tag} ≠ {answer} := by
  try rfl
  try norm_num"""

ANSWER_TACTIC = """
example: {answer_tag} = {answer} := by sorry"""

AUTO_TACTIC = """example: {answer_tag} = {answer} := by {tactic}"""


class ErrorType(StrEnum):
    SUCCESS = "The answer and the proof are both correct."
    GENERATION_ERROR = "Uncaught error in the model generation."
    GENERATION_TIMEOUT = "Model generation timeout."
    GENERATION_REPEAT = "Model generation repeat."
    GENERATION_TOO_LONG = "Model generation too long."
    FORMAT_ERROR = "Catch format error in the Lean 4 code."
    ANSWER_NOT_MATCHED = "Can not find all answers in the lean code."
    WRONG_ANSWERS = (
        "Failed to prove that the answer is equal to the ground truth by 'rfl'."
    )
    PROOF_FAILED = "The proof failed."
    ANSWER_PROOF_FAILED = (
        "Failed to prove that the answer is equal to the ground truth."
    )


class ProofStage(StrEnum):
    PROOF = "proof"
    SIMP = "simp"
    NEGATION = "negation"
    ANSWER = "answer"


class LLMClientType(StrEnum):
    OpenAI = "OpenAI"
    Claude = "Claude"
    Gemini = "Gemini"
    TogetherAI = "TogetherAI"


@dataclass
class GenerationTask:
    problem_id: str | UUID
    messages: list[str, str]
    formal_statement: str | list[str]
    stage: ProofStage = ProofStage.PROOF
    ground_truths: list[str] | None = None


@dataclass
class VerificationTask:
    problem_id: str
    text: str
    formal_statement: str | list[str]
    stage: ProofStage = ProofStage.PROOF
    ground_truths: list[str] | None = None


@dataclass
class VerificationResult:
    problem_id: str | UUID
    error_type: ErrorType
    raw_text: str
    stage: ProofStage = ProofStage.PROOF
    lean4_code: str | None = None
    answer_predicts: list[str] | None = None
    lean_feedback: dict | None = None


@dataclass
class EvaluationResult:
    problem_id: str | UUID
    formal_statement: str
    ground_truths: list[str] | None = None
    query_messages: list[dict[str, str]] | None = None
    responses: list[str] | None = None
    solution_attempts: list[str] | None = None
    lean_feedbacks: list[dict] | None = None
    error_types: list[ErrorType] | None = None
    passed_at: list[int] | None = None
    correct_solutions: list[str] | None = None
    one_success_solution: str | None = None
