import Mathlib

abbrev brualdi_ch7_15_solution : PowerSeries ℝ := sorry

/--
Determine the generating function for the sequence of cubes \[ 0, 1, 8, \ldots, n^{3}, \ldots \]
-/
theorem brualdi_ch7_15 : PowerSeries.mk (fun (n : ℕ) => (n : ℝ) ^ 3) = brualdi_ch7_15_solution := by
    sorry
