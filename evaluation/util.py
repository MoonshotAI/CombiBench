import re
import sys
from datetime import datetime

import loguru

# Configure logger with file and line information for clickable links
logger = loguru.logger
logger.remove()  # Remove default handler
logger.add(
    sink=sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}:{line}</cyan> | <cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
    level="DEBUG",
)
logger = logger.bind(name="Easy Verify Lean 4 Proof")


DEFAULT_HEADERS = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""


def get_datetime():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def remove_comments(text: str) -> str:
    # First remove multi-line comments
    text = re.sub(r"/-[\s\S]*?-/", "", text)
    # Then remove single-line comments
    return re.sub(r"^\s*--.*\n", "\n", text, flags=re.MULTILINE)


def remove_sorry_statements(text: str) -> str:
    return text.replace("sorry", "").strip()


def add_sorry_to_statement(statement: str) -> str:
    """
    Adds a "sorry" to the end of the statement if it doesn't already have one.
    """
    statement = statement.rstrip()
    # Check if the statement ends with "sorry"
    if not statement.endswith("sorry"):
        # Add "sorry" to the end of the statement
        statement += " sorry"
    return statement


def extract_formal_statement_from_text(text: str) -> str | None:
    formal_statements = re.findall(r"```formal_statement\n(.*?)\n```", text, re.DOTALL)
    if formal_statements:
        return formal_statements[-1].strip()
    return None


def extract_code_and_answer_from_text(
    text: str,
    formal_statements: list[str] | str,
    allow_no_header: bool = False,
) -> tuple[str | None, dict[str, str] | None]:
    """
    Extracts a proof from a Lean 4 code block, ensuring it follows the formal statements.
    """

    if isinstance(formal_statements, str):
        formal_statements = remove_comments(formal_statements).split("\n\n")

    formal_statements = [
        fs.replace("\ntheorem", "\n\ntheorem") for fs in formal_statements
    ]
    formal_statements = [
        remove_sorry_statements(fs).strip()
        for fs in formal_statements
        if not fs.startswith(("import", "set_option", "open"))
    ]

    # Extract all Lean 4 code blocks
    lean4_codes = re.findall(r"```lean4\n(.*?)\n```", text, re.DOTALL)

    if len(lean4_codes) == 0:
        # a looser detection
        lean4_codes = re.findall(r"```lean\n(.*?)\n```", text, re.DOTALL)
        if len(lean4_codes) != 0:
            logger.warning("No lean4 code found, using the last lean code")
        else:
            return None, None

    lean4_code = remove_comments(lean4_codes[-1]).strip()

    if allow_no_header and not lean4_code.startswith("import"):
        lean4_code = DEFAULT_HEADERS + lean4_code

    if "axiom" in lean4_code or "local_instance" in lean4_code:
        logger.debug("'axiom' or 'local_instance' is not allowed.")
        return None, None

    if not all(
        formal_statement in lean4_code for formal_statement in formal_statements
    ):
        return None, None

    answers = {}
    for formal_statement in formal_statements:
        if "_solution" in formal_statement and "abbrev" in formal_statement:
            try:
                answer_tag = (
                    formal_statement.split("abbrev")[1].split("_solution")[0].strip()
                )
                answer = formal_statement.split(":=")[-1].strip()
            except (IndexError, ValueError):
                return lean4_code, None
            answers[f"{answer_tag}_solution"] = answer

    return lean4_code, answers


def extract_theorem_name_from_code(code: str) -> str:
    # Regular expression to match theorem or lemma names after the keywords `theorem` or `lemma`
    pattern = r"(?:theorem|lemma)\s+(\w+)"

    # Find all matches for theorem or lemma names
    theorem_names = re.findall(pattern, code)

    return theorem_names[-1]


def extrac_first_tactic_block(code: str) -> str:
    pattern = r"```tactics\n(.*?)\n```"
    tactic_blocks = re.findall(pattern, code, re.DOTALL)
    if tactic_blocks:
        return tactic_blocks[0].strip()
    return None


def generate_formal_statement(example):
    prove_formal_statement = "import Mathlib -- firstly generate the autoformalized statement using the name `{theorem_name}` and then prove it."
    fb_formal_statement = "import Mathlib -- It's a fill-in-the-blank problem, firstly write a line to indicate the type of the answer using `abbrev {theorem_name}_solution : <type> := sorry`, and formalize the statement using the name `{theorem_name}` and then prove it."
    if example.get("type") == "word":
        return fb_formal_statement.format(theorem_name=example["theorem_name"])
    else:
        return prove_formal_statement.format(theorem_name=example["theorem_name"])


if __name__ == "__main__":
    text = "### Detailed Proof and Analysis\n\nWe are given:\n1. `b`, `h`, and `v` are positive real numbers.\n2. The volume of a cone is given by `v = (1/3) * b * h`.\n3. The base area `b = 30`.\n4. The height `h = 13/2 = 6.5`.\nWe need to prove that `v = 65`.\n\n#### Step 1: Substitute the Given Values into the Volume Formula\nFirst, substitute `b = 30` and `h = 13/2` into the formula `v = (1/3) * b * h`:\n```\nv = (1/3) * 30 * (13/2)\n```\n\n#### Step 2: Simplify the Expression\nSimplify the expression step by step:\n1. Multiply `(1/3)` and `30`:\n   ```\n   (1/3) * 30 = 10\n   ```\n2. Multiply the result by `13/2`:\n   ```\n   10 * (13/2) = (10 * 13)/2 = 130/2 = 65\n   ```\nThus, `v = 65`.\n\n#### Step 3: Formal Verification\nIn Lean, we can directly substitute the values and simplify using arithmetic operations. We do not need any additional lemmas since the problem is purely computational.\n\n### Step-by-Step Abstract Plan\n\n1. **Substitute `b = 30` and `h = 13/2` into the equation `v = (1/3) * b * h`.**\n2. **Calculate `(1/3) * 30` to get `10`.**\n3. **Calculate `10 * (13/2)` to get `65`.**\n4. **Conclude that `v = 65`.**\n\n### Lean 4 Proof with `have` Statements\n\n```lean4\ntheorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))\n    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by\n  have h₄ : v = 65 := by sorry\n  sorry\n```\n\n### Complete Lean 4 Proof\n\n```lean4\ntheorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))\n    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by\n  have h₄ : v = 65 := by\n    have h₅ : v = 1 / 3 * (b * h) := h₁\n    rw [h₅]\n    have h₆ : b = 30 := h₂\n    have h₇ : h = 13 / 2 := h₃\n    rw [h₆, h₇]\n    norm_num\n    <;>\n    (try norm_num at h₀ ⊢) <;>\n    (try linarith) <;>\n    (try ring_nf at h₀ ⊢) <;>\n    (try norm_num at h₀ ⊢) <;>\n    (try linarith)\n  \n  exact h₄\n```"
    formal_statement = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n/-- The volume of a cone is given by the formula $V = \\frac{1}{3}Bh$, where $B$ is the area of the base and $h$ is the height. The area of the base of a cone is 30 square units, and its height is 6.5 units. What is the number of cubic units in its volume? Show that it is 65.-/\ntheorem mathd_algebra_478 (b h v : ℝ) (h₀ : 0 < b ∧ 0 < h ∧ 0 < v) (h₁ : v = 1 / 3 * (b * h))\n    (h₂ : b = 30) (h₃ : h = 13 / 2) : v = 65 := by"
    code, _ = extract_code_and_answer_from_text(
        text, formal_statement, allow_no_header=True
    )
    print(code)
