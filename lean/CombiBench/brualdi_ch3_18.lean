import Mathlib

/--
Prove that of any five points chosen within a square of side length 2 , there are two whose distance apart is at most $\sqrt{2}$.
-/
theorem brualdi_ch3_18
    (points : Fin 5 → (EuclideanSpace ℝ (Fin 2)))
    (h_points : ∀ i, 0 ≤ ((points i) 0) ∧ ((points i) 0) ≤ 2 ∧ 0 ≤ ((points i) 1) ∧ ((points i) 1) ≤ 2) :
    ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ √2 := by sorry
