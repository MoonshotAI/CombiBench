import Mathlib

open Finset

inductive Letters
  | I | M | O
deriving BEq, DecidableEq

def diagonal_I_set (n k : ℕ) : Finset (Fin n × Fin n) :=
  (@Finset.univ (Fin n × Fin n) _ |>.filter (fun (i, j) => i.1 + j.1 = k))

def diagonal_I (n k : ℕ) (A : Matrix (Fin n) (Fin n) Letters) : Prop :=
  3 ∣ #(diagonal_I_set n k) →
  #(diagonal_I_set n k) = 3 * #{x ∈ diagonal_I_set n k | A x.1 x.2 = Letters.I} ∧
  #(diagonal_I_set n k) = 3 * #{x ∈ diagonal_I_set n k | A x.1 x.2 = Letters.M} ∧
  #(diagonal_I_set n k) = 3 * #{x ∈ diagonal_I_set n k | A x.1 x.2 = Letters.O}

def diagonal_II_set (n : ℕ) (k : ℤ) : Finset (Fin n × Fin n) :=
  (@Finset.univ (Fin n × Fin n) _ |>.filter (fun (i, j) => (i.1 : ℤ) - (j.1 : ℤ) = k))

def diagonal_II (n : ℕ) (k : ℤ) (A : Matrix (Fin n) (Fin n) Letters) : Prop :=
  3 ∣ #(diagonal_II_set n k) →
  #(diagonal_II_set n k) = 3 * #{x ∈ diagonal_II_set n k | A x.1 x.2 = Letters.I} ∧
  #(diagonal_II_set n k) = 3 * #{x ∈ diagonal_II_set n k | A x.1 x.2 = Letters.M} ∧
  #(diagonal_II_set n k) = 3 * #{x ∈ diagonal_II_set n k | A x.1 x.2 = Letters.O}

def exists_valid_table (n : ℕ) : Prop :=
  ∃ (A : Matrix (Fin n) (Fin n) Letters),
  (∀ i : Fin n,
    n = 3 * #{j | A i j = Letters.I} ∧ n = 3 * #{j | A i j = Letters.M} ∧
    n = 3 * #{j | A i j = Letters.O}) ∧
  (∀ j : Fin n,
    n = 3 * #{i | A i j = Letters.I} ∧ n = 3 * #{i | A i j = Letters.M} ∧
    n = 3 * #{i | A i j = Letters.O}) ∧
  (∀ k ∈ Finset.range (2 * n - 1), diagonal_I n k A) ∧
  ∀ k ∈ Finset.Icc (-(n : ℤ) + 1) (n - 1), diagonal_II n k A

abbrev imo_2016_p2_solution : Set ℕ := sorry

/--
Find all integers $n$ for which each cell of $n \times n$ table can be filled with one of the letters $I,M$ and $O$ in such a way that: in each row and each column, one third of the entries are $I$, one third are $M$ and one third are $O$; and in any diagonal, if the number of entries on the diagonal is a multiple of three, then one third of the entries are $I$, one third are $M$ and one third are $O$. Note. The rows and columns of an $n \times n$ table are each labelled $1$ to $n$ in a natural order. Thus each cell corresponds to a pair of positive integer $(i,j)$ with $1 \le i,j \le n$. For $n>1$, the table has $4n-2$ diagonals of two types. A diagonal of first type consists all cells $(i,j)$ for which $i+j$ is a constant, and the diagonal of this second type consists all cells $(i,j)$ for which $i-j$ is constant.
-/
theorem imo_2016_p2 : {n | exists_valid_table n} = imo_2016_p2_solution := by sorry
