import Mathlib

abbrev hackmath_9_solution : ℕ := sorry

/--
The father has six sons and ten identical, indistinguishable balls. How many ways can he give the balls to his sons if everyone gets at least one?
-/
theorem hackmath_9 (sols : Finset (Fin 6 → ℕ))
    (h_sols : ∀ f, f ∈ sols ↔ ((∀ i, f i > 0) ∧ (∑ i, f i = 10))) :
    sols.card = hackmath_9_solution := by sorry
