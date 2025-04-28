import Mathlib

open Finset

abbrev brualdi_ch2_11_solution : ℕ := sorry

/--
How many sets of three integers between 1 and 20 are possible if no two consecutive integers are to be in a set?
-/
theorem brualdi_ch2_11 :
    ((Icc (1 : ℕ) 20).powersetCard 3 |>.filter (fun S => ∀ a ∈ S, a - 1 ∉ S ∧ a + 1 ∉ S)).card =
    brualdi_ch2_11_solution := by sorry
