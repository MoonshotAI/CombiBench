import Mathlib

structure goodPairs (s : Fin 100 ↪ ℕ+) where
    (i j : Fin 100)
    (ratio : s i = 2 * s j ∨ s i = 3 * s j)
deriving Fintype

abbrev izho_2014_p3_solution : ℕ := sorry

/--
There are given 100 distinct positive integers. We call a pair of integers among them good if the ratio of its elements is either 2 or 3. What is the maximum number $g$ of good pairs that these 100 numbers can form? (A same number can be used in several pairs.)
-/
theorem izho_2014_p3 :
    IsGreatest (Set.range fun x => Fintype.card (goodPairs x)) izho_2014_p3_solution := by sorry
