import Mathlib

structure Index (n : ℕ+) where
  row : ℕ
  col : ℕ
  -- counting from 1
  le_row : 1 ≤ row
  row_le : row ≤ n
  -- counting from 1
  le_col : 1 ≤ col
  col_le : col ≤ row

--     x
--   q x
-- p x x
def Index.atBottomLeft {n : ℕ+} (p q : Index n) : Prop :=
  q.row + 1 = p.row ∧ q.col = p.col

--     x
--   q x
-- x p x
def Index.atBottomRight {n : ℕ+} (p q : Index n) : Prop :=
  q.row + 1 = p.row ∧ q.col + 1 = p.col

@[simps]
def triangleGraph (n : ℕ+) : Digraph (Index n) where
  Adj p q :=
    p.atBottomLeft q ∨ p.atBottomRight q ∨
    q.atBottomLeft p ∨ q.atBottomRight p

-- each row has one red circle
abbrev JapaneseTriangle (n : ℕ+) := ∀ (i : Fin n), Fin (i + 1)

structure NinjaPath {n : ℕ+} (jt : JapaneseTriangle n) where
  path : RelSeries (triangleGraph n |>.Adj)
  length : path.length = n.natPred

@[simp]
lemma NinjaPath.path_length_succ {n : ℕ+} {jt : JapaneseTriangle n} (p : NinjaPath jt) :
    p.path.length + 1 = n := by
  simp [p.length]

def NinjaPath.countRed {n : ℕ+} {jt : JapaneseTriangle n} (p : NinjaPath jt) : ℕ :=
  ∑ (i : Fin (p.path.length + 1)),
    if (jt (Fin.cast (by simp) i) : ℕ) = (p.path i).col
    then 1
    else 0

noncomputable abbrev imo_2023_p5_solution : ℕ+ → ℕ := sorry

/--
Let $n$ be a positive integer. A Japanese triangle consists of $1 + 2 + \dots + n$ circles arranged in an equilateral triangular shape such that for each $i = 1$, $2$, $\dots$, $n$, the $i^{th}$ row contains exactly $i$ circles, exactly one of which is coloured red. A ninja path in a Japanese triangle is a sequence of $n$ circles obtained by starting in the top row, then repeatedly going from a circle to one of the two circles immediately below it and finishing in the bottom row. In terms of $n$, find the greatest $k$ such that in each Japanese triangle there is a ninja path containing at least $k$ red circles.
-/
theorem imo_2023_p5 (n : ℕ+) :
    IsGreatest {k | ∀ (jt : JapaneseTriangle n), ∃ (p : NinjaPath jt), k ≤ p.countRed}
    (imo_2023_p5_solution n) := by sorry
