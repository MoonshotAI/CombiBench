import Mathlib

def invNum {n : ℕ} (σ : Equiv.Perm (Fin n)) : ℕ :=
  ∑ x ∈ Equiv.Perm.finPairsLT n, if σ x.fst ≤ σ x.snd then 0 else 1

/--
Show that the largest number of inversions of a permutation of ${1, 2, ... , n}$ equals $\frac{n(n -1)}{2}$.
-/
theorem brualdi_ch4_9 (n : ℕ) :
    IsGreatest {k | ∃ σ : Equiv.Perm (Fin n), k = invNum σ} (n * (n - 1) / 2) := by sorry
