import Mathlib

def appears (W : ℤ → Fin 2) (U : Σ n, Fin n → Fin 2) : Prop :=
  ∃ k, ∀ i : Fin U.1, U.2 i = W (k + i)

def ubiquitous (W : ℤ → Fin 2) (U : Σ n, Fin n → Fin 2) : Prop :=
  appears W ⟨U.1 + 1, Fin.snoc U.2 0⟩ ∧ -- U a
  appears W ⟨U.1 + 1, Fin.snoc U.2 1⟩ ∧ -- U b
  appears W ⟨U.1 + 1, Fin.cons 0 U.2⟩ ∧ -- a U
  appears W ⟨U.1 + 1, Fin.cons 1 U.2⟩ -- b U

/--
Let $n$ be a positive integer and let $W=\ldots x_{-1} x_{0} x_{1} x_{2} \ldots$ be an infinite periodic word consisting of the letters $a$ and $b$. Suppose that the minimal period $N$ of $W$ is greater than $2^{n}$. A finite nonempty word $U$ is said to appear in $W$ if there exist indices $k \leq \ell$ such that $U=x_{k} x_{k+1} \ldots x_{\ell}$. A finite word $U$ is called ubiquitous if the four words $U a, U b, a U$, and $b U$ all appear in $W$. Prove that there are at least $n$ ubiquitous finite nonempty words.
-/
theorem imosl_2011_c6 (W : ℤ → Fin 2) (n : ℕ+) (N : ℕ) (hN : 2 ^ n.1 < N)
    (hW : Function.Periodic W N) (hW' : ∀ N' < N, ¬ Function.Periodic W N') :
    ∃ (x : Fin n ↪ (Σ k, Fin k → Fin 2)), (∀ i, (x i).1 ≠ 0) ∧ (∀ i, ubiquitous W (x i)) := by sorry
