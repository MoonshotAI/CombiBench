import Mathlib

def invNum {n : ℕ} (σ : Equiv.Perm (Fin n)) : ℕ :=
  ∑ x ∈ Equiv.Perm.finPairsLT n, if σ x.fst ≤ σ x.snd then 0 else 1

/--
Let $n \geq 2$ be an integer. Prove that the total number of inversions of all $n$ ! permutations of $1,2, \ldots, n$ equals $\frac{1}{2} n!\binom{n}{2}=n!\frac{n(n-1)}{4}$ (Hint: Pair up the permutations so that the number of inversions in each pair is $\frac{n(n-1)}{2}$.)
-/
theorem brualdi_ch4_59 (n : ℕ) (hn : n ≥ 2) : ∑ σ : Equiv.Perm (Fin n), invNum σ =
    n.factorial * n * (n - 1) / 4 := by sorry
