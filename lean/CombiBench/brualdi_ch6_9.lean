import Mathlib

open Finset

abbrev brualdi_ch6_9_solution : ℕ := sorry

/--
Determine the number of integral solutions of the equation $x_{1}+x_{2}+x_{3}+x_{4}=20$ that satisfy $1 \leq x_{1} \leq 6,0 \leq x_{2} \leq 7,4 \leq x_{3} \leq 8,2 \leq x_{4} \leq 6$.
-/
theorem brualdi_ch6_9 : {x : Fin 4 → ℕ | x 0 ∈ Icc 1 6 ∧ x 1 ∈ Icc 0 7 ∧
    x 2 ∈ Icc 4 8 ∧ x 3 ∈ Icc 2 6}.ncard = brualdi_ch6_9_solution := by sorry
