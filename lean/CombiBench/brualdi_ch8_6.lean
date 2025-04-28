import Mathlib

abbrev brualdi_ch8_6_solution : ℕ → ℝ := sorry

/--
Let the sequence $h_{0}, h_{1}, \ldots, h_{n}, \ldots$ be defined by $h_{n}=2 n^{2}-n+3,(n \geq 0)$. Find a formula for $\sum_{k=0}^{n} h_{k}$.
-/
theorem brualdi_ch8_6 (n : ℕ) (h : ℕ → ℝ) (h' : ∀ i, h i = 2 * i ^ 2 - i + 3) :
    ∑ i ∈ Finset.range (n + 1), h i = brualdi_ch8_6_solution n := by sorry
