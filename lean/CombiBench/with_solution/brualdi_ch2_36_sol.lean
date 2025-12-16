import Mathlib


/--
Determine the total number of combinations (of any size) of a multiset of objects of $k$ different types with finite repetition numbers $n_{1}, n_{2}, \ldots, n_{k}$, respectively.
-/
theorem brualdi_ch2_36 {k : ℕ} (n : Fin k → ℕ)
    (sols : Finset (Fin k → ℕ))
    (h_sols : ∀ f, f ∈ sols ↔ (∀ i, f i ≤ n i)) :
    sols.card = ((fun n => (∏ i : Fin k, (n i + 1))) : (Fin k → ℕ) → ℕ ) n := by sorry
