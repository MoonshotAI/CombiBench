import Mathlib

structure PascalIndex (n : ℕ) where
  (i : ℕ) (j : ℕ)
  (i_le_n : i ≤ n)
  (j_le_i : j < i)

def IsAntiPascal (n : ℕ) (values : PascalIndex n → ℤ) : Prop :=
  ∀ index : PascalIndex n,
    if h : index.i < n then
      values index =
        abs (
          values { i := index.i + 1, j := index.j, i_le_n := by omega, j_le_i := by linarith [index.j_le_i] } -
          - values { i := index.i + 1, j := index.j + 1, i_le_n := by omega, j_le_i := by linarith [index.j_le_i] }
        )
    else
      true


/--
An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from $1$ to $10$ \[4\]\[2\quad 6\]\[5\quad 7 \quad 1\] \[8\quad 3 \quad 10 \quad 9\] Does there exist an anti-Pascal triangle with $2018$ rows which contains every integer from $1$ to $1 + 2 + 3 + \dots + 2018$?
-/
theorem imo_2018_p3 : ((false) : Bool ) = ∃ values, IsAntiPascal 2018 values ∧
    Finset.Icc (1 : ℤ) (∑ i ∈ Finset.Icc 1 2018, i) = {x | ∃ i, x = values i} := by
    sorry
