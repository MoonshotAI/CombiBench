import Mathlib

/--
Prove that of any five points chosen within a square of side length 2 , there are two whose distance apart is at most $\sqrt{2}$.
-/
theorem brualdi_ch3_18
    (points : Fin 5 → Set.Icc (0 : ℝ) 2 × Set.Icc (0 : ℝ) 2) :
    ∃ i j, i ≠ j ∧ Dist.dist (points i) (points j) ≤ √2 := by sorry
