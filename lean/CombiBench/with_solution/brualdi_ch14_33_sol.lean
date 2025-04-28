import Mathlib

/--
Prove that a permutation and its inverse have the same type.
-/
theorem brualdi_ch14_33 {α : Type*} [Fintype α] [DecidableEq α] (σ : Equiv.Perm α) :
    σ.cycleType = σ⁻¹.cycleType := by sorry
