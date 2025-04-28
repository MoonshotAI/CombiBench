import Mathlib

structure IsMagicSquare {n : ℕ} (M : Matrix (Fin n) (Fin n) ℕ) : Prop where
  mem : ∀ i j, M i j ∈ Finset.Icc 1 (n * n)
  pairwise : ∀ i j i' j', i ≠ i' ∨ j ≠ j' → M i j ≠ M i' j'
  same_sum : ∃ s, (∀ i, ∑ j, M i j = s) ∧ (∀ j, ∑ i, M i j = s) ∧ (∑ i, M i i.rev = s) ∧ ∑ i, M i i = s

abbrev replace {n : ℕ}: Matrix (Fin n) (Fin n) ℕ → Matrix (Fin n) (Fin n) ℕ :=
  fun A i j ↦ n^2 + 1 - A i j

/--
Show that the result of replacing every integer a in a magic square of order n with $n^2 + 1 − a$ is a magic square of order n.
-/
theorem brualdi_ch1_16 {n : ℕ} (M : Matrix (Fin n) (Fin n) ℕ) (hM : IsMagicSquare M) :
    IsMagicSquare (replace M) := by sorry
