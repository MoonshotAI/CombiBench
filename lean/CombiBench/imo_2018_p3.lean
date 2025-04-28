import Mathlib

def IsAntiPascal (L : List (List ℕ)) : Prop :=
  L ≠ [] ∧ L[0]!.length = L.length ∧ (∀ i, i < L.length - 1 → (L[i]!.length = L[i+1]!.length + 1 ∧
  ∀ j, j < (L[i+1]!).length → ((L[i+1]!)[j]! = (((L[i]!)[j]! : ℤ) - (L[i]!)[j+1]!).natAbs)))

abbrev imo_2018_p3_solution : Bool := sorry

/--
An anti-Pascal triangle is an equilateral triangular array of numbers such that, except for the numbers in the bottom row, each number is the absolute value of the difference of the two numbers immediately below it. For example, the following is an anti-Pascal triangle with four rows which contains every integer from $1$ to $10$ \[4\]\[2\quad 6\]\[5\quad 7 \quad 1\] \[8\quad 3 \quad 10 \quad 9\] Does there exist an anti-Pascal triangle with $2018$ rows which contains every integer from $1$ to $1 + 2 + 3 + \dots + 2018$?
-/
theorem imo_2018_p3 : imo_2018_p3_solution = ∃ l, IsAntiPascal l ∧
    List.range' 1 (∑ i ∈ Finset.Icc 1 2018, i) = List.insertionSort (· ≤ ·) (List.flatten l) := by
    sorry
