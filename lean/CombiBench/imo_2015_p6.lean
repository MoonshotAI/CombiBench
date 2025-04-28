import Mathlib

/--
The sequence $a_1,a_2,\dots$ of integers satisfies the conditions: (i) $1\le a_j\le2015$ for all $j\ge1$, (ii) $k+a_k\neq \ell+a_\ell$ for all $1\le k<\ell$. Prove that there exist two positive integers $b$ and $N$ for which\[\left\vert\sum_{j=m+1}^n(a_j-b)\right\vert\le1007^2\]for all integers $m$ and $n$ such that $n>m\ge N$.
-/
theorem imo_2015_p6 (a : ℕ+ → ℤ) (ha1 : ∀ j : ℕ+, 1 ≤ a j ∧ a j ≤ 2015)
    (ha2 : ∀ k l, k < l → k + a k ≠ l + a l) :
    ∃ b N : ℕ+, ∀ m n, n > m ∧ m ≥ N → |(∑ j ∈ Finset.Icc (m + 1) n, (a j - b))| ≤ 1007^2 := by sorry
