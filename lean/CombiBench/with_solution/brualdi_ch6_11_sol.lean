import Mathlib


/--
Determine the number of permutations of $\{1,2, \ldots, 8\}$ in which no even integer is in its natural position.
-/
theorem brualdi_ch6_11
    (sols : Finset (Equiv.Perm (Finset.Icc 1 8)))
    (h_sols : ∀ σ, σ ∈ sols ↔ (∀ i, Even i.1 → σ i ≠ i)) :
    sols.card = ((216) : ℕ ) := by sorry
