import re
from datetime import datetime

import loguru


logger = loguru.logger.bind(name='Fill-in-the-Blank in Lean')


def get_datetime():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def remove_comments(text: str) -> str:
    # First remove multi-line comments
    text = re.sub(r'/-[\s\S]*?-/', '', text)
    # Then remove single-line comments
    return re.sub(r'^\s*--.*\n', '\n', text, flags=re.MULTILINE)


def remove_sorry_statements(text: str) -> str:
    return re.sub(r'sorry\s*', '', text).strip()


def extract_code_and_answer_from_text(
    text: str, formal_statements: list[str] | str,
) -> tuple[str | None, dict[str, str] | None]:
    """
    Extracts a proof from a Lean 4 code block, ensuring it follows the formal statements.
    """

    if isinstance(formal_statements, str):
        formal_statements = formal_statements.split('\n\n')

    # Extract all Lean 4 code blocks
    lean4_codes = re.findall(r'```lean4\n(.*?)\n```', text, re.DOTALL)

    if len(lean4_codes) == 0:
        # a looser detection
        lean4_codes = re.findall(r'```lean\n(.*?)\n```', text, re.DOTALL)
        if len(lean4_codes) != 0:
            logger.warning('No lean4 code found, using the last lean code')
        else:
            return None, None

    lean4_code = lean4_codes[-1].strip()
    lean4_code = remove_comments(lean4_code)

    if not all(formal_statement in lean4_code for formal_statement in formal_statements):
        logger.warning('The formal statements are not found in the Lean 4 code.')
        return None, None

    if 'axiom' in lean4_code:
        logger.debug("'axiom' is not allowed.")
        return None, None

    answers = {}
    for formal_statement in formal_statements:
        if '_solution' in formal_statement and 'abbrev' in formal_statement:
            try:
                answer_tag = formal_statement.split('abbrev')[1].split('_solution')[0].strip()
                answer = formal_statement.split(':=')[-1].strip()
            except (IndexError, ValueError):
                return lean4_code, None
            answers[f'{answer_tag}_solution'] = answer

    return lean4_code, answers
