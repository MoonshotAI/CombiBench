import Mathlib

/--
Show that if $n+1$ integers are chosen from the set ${1,2, \ldots, 2 n}$, then there are always two which differ by 1.
-/
theorem brualdi_ch3_4 (n : ℕ) (S : Finset ℕ) (elem_range : ∀ s ∈ S, (1 ≤ s ∧ s ≤ 2 * n))
    (card : S.card = n + 1) : ∃ s ∈ S, ∃ s' ∈ S, s = s' + 1 := by sorry
