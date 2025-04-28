import Mathlib

/--
Let $n$ be an odd prime number. Prove that each of the permutations, $\rho_{n}, \rho_{n}^{2}, \ldots, \rho_{n}^{n}$ of $\{1,2, \ldots, n\}$ is an $n$-cycle. (Recall that $\rho_{n}$ is the permutation that sends 1 to 2,2 to $3, \ldots, n-1$ to $n$, and $n$ to 1.)
-/
theorem brualdi_ch14_45 {n : ℕ} (h : Odd n) (hp : Nat.Prime n) :
    ∀ i ∈ Finset.Icc 1 n, ((finRotate n) ^ i).IsCycle := by sorry
