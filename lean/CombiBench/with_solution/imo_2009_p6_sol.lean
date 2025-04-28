import Mathlib

open BigOperators Finset

/--
Let $a_1,a_2,\ldots,a_n$ be distinct positive integers and let $M$ be a set of $n-1$ positive integers not containing $s=a_1+a_2+\ldots+a_n$. A grasshopper is to jump along the real axis, starting at the point $0$ and making $n$ jumps to the right with lengths $a_1,a_2,\ldots,a_n$ in some order. Prove that the order can be chosen in such a way that the grasshopper never lands on any point in $M$.
-/
theorem imo_2009_p6 (n : ℕ) (hn : n ≥ 1) (a : Fin n → ℕ) (ha : Function.Injective a) (M : Finset ℕ)
    (ha' : ∀ i, a i > 0) (hM : M.card = n - 1) (hM' : ∀ m ∈ M, m > 0) (haM : ∑ n, (a n) ∉ M) :
    ∃ (σ : Equiv.Perm (Fin n)), ∀ k, (∑ i ≤ k, (a ∘ σ) i) ∉ M := by sorry
