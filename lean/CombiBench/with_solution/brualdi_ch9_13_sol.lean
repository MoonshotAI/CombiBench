import Mathlib

/--
Let $A$ be a matrix with $n$ columns, with integer entries taken from the set $S=\{1,2, \ldots, k\}$. Assume that each integer $i$ in $S$ occurs exactly $n r_{i}$ times in $A$, where $r_{i}$ is an integer. Prove that it is possible to permute the entries in each row of $A$ to obtain a matrix $B$ in which each integer $i$ in $S$ appears $r_{i}$ times in each column.
-/
theorem brualdi_ch9_13 (n m k : ℕ) (r : ℕ → ℕ) (A : Matrix (Fin m) (Fin n) ℕ)
    (hn : n > 0) (hm : m > 0)(hk : k ≥ 1)
    (hA : ∀ i j, A i j ∈ Finset.Icc 1 k)
    (hr : ∀ i ∈ Finset.Icc 1 k, (∑ x : Fin m, ∑ y : Fin n, if A x y = i then 1 else 0) = n * r i) :
    ∃ (rσ : Fin m → Equiv.Perm (Fin n)),
      ∀ j : Fin n, ∀ i ∈ Finset.Icc 1 k,
      (∑ x : Fin m, if A x ((rσ x) j) = i then 1 else 0) = r i := by sorry
