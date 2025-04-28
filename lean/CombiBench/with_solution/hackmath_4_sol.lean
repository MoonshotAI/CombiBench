import Mathlib


/--
How many people must be in a group for at least two of them to be born in the same month?
-/
theorem hackmath_4 : IsLeast {n | ∀ f : Fin n → Fin 12, ∃ a b, f a = f b} ((13) : ℕ ) := by sorry
