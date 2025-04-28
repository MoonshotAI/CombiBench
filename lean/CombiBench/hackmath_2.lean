import Mathlib

abbrev hackmath_2_solution : ℕ := sorry

/--
There are 8 athletes participating in a sprint competition. The referee needs to select 3 athletes and assign them specific rankings (first place, second place, and third place). How many different arrangements are possible?
-/
theorem hackmath_2 (sols : Finset (Fin 8 → Fin 4))
    (h_sols : ∀ f, f ∈ sols ↔
      ((List.ofFn f).count 0 = 1) ∧ ((List.ofFn f).count 1 = 1) ∧ ((List.ofFn f).count 2 = 1)) :
    sols.card = hackmath_2_solution := by sorry
