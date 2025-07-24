import nest_asyncio
from evaluation.client.lean_client import Lean4Client
import os

# Enable nested asyncio for Jupyter notebooks
nest_asyncio.apply()

client = Lean4Client(base_url="http://35.202.160.43", api_key=os.getenv("NUMINA_SERVER_API_KEY"))
mock_proof = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

theorem lean_workbook_10009 (a b c: ℝ) (ha : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1): a^3 + b^3 + c^3 + (15 * a * b * c)/4 ≥ 1/4 := by

nlinarith [sq_nonneg (a - b), sq_nonneg (b - c), sq_nonneg (c - a),
sq_nonneg (a + b + c)]"""

response = client.verify([
    {"proof": mock_proof, "custom_id": "1"},
    {"proof": mock_proof, "custom_id": "2"}
], timeout=30)
