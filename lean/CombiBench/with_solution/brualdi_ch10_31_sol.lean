import Mathlib

def isDifferenceSet (n : ℕ) (B : Finset (ZMod n)) : Prop :=
  ∃ k, ∀ x : (ZMod n),  x ≠ 0 → ∑ i ∈ B, ∑ j ∈ B \ {i}, List.count x [i - j] = k

/--
Prove that $B = {0,3,4,9,11}$ is a difference set in $Z_{21}$.
-/
theorem brualdi_ch10_31 : isDifferenceSet 21 {0, 3, 4, 9, 11} := by sorry
