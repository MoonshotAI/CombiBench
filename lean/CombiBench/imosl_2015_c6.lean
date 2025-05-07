import Mathlib

def clean (S : Set ℕ+) (n : ℕ) : Prop :=
  ∃! (S' : Finset ℕ+),
    ((S' : Set _) ⊆ S) ∧ (Odd S'.card) ∧ (∑ s ∈ S', (s : ℕ) = n)

/--
Let $S$ be a nonempty set of positive integers. We say that a positive integer $n$ is clean if it has a unique representation as a sum of an odd number of distinct elements from $S$. Prove that there exist infinitely many positive integers that are not clean.
-/
theorem imosl_2015_c6 (S : Set ℕ+) (hS : S.Nonempty): ∀ (N : ℕ), ∃ (m : ℕ), N < m ∧ ¬ clean S m := by sorry
