import Mathlib

open scoped Classical Finset

instance {N : ℕ} : CircularOrder (Fin N) := LinearOrder.toCircularOrder _

variable {α : Type*} [CircularOrder α] {a b c d : α}

-- In a circular order, the property that `a, b, c, d` are in that order.
def SBtw₄ (a b c d : α) : Prop := sbtw a b c ∧ sbtw c d a

lemma sbtw₄_swap : SBtw₄ a b c d ↔ SBtw₄ c d a b := and_comm

variable {N : ℕ}

-- The diagonals of the `N`-gon.
abbrev Diagonal (N : ℕ) := {e : Sym2 (Fin N) // ¬ e.IsDiag}

namespace Diagonal

def Intersect (d₁ d₂ : Diagonal N) : Prop :=
  Sym2.lift₂ {
    val a b c d := SBtw₄ a c b d ∨ SBtw₄ a d b c
    property a b c d := by
      simp only [eq_iff_iff]; constructor <;> rw [sbtw₄_swap, or_comm, sbtw₄_swap]
  } d₁.1 d₂.1

def Good (d : Diagonal N) : Prop :=
  Sym2.lift {
    val a b := Odd (a.val + b.val : ℕ)
    property a b := by simp [add_comm]
  } d.1

end Diagonal

-- if a collection of `2N - 3` distinct diagonal are pairwise non-intersecting, they dissect the
-- N-gon
structure TriangleDissection (N : ℕ) where
  diagonals : Fin (2 * N - 3) ↪ Diagonal N
  pairwise_not_intersect_diagonals : Pairwise fun i j ↦ ¬ (diagonals i).Intersect (diagonals j)

noncomputable def TriangleDissection.numOfIsoscelesTriangle (C : TriangleDissection N) : ℕ := by
  classical exact
  #{(a, b, c) : Fin N × Fin N × Fin N |
    ∃ (hab : a < b) (hbc : b < c),
      -- The edges of the triangle belong to the diagonals
      (∃ i, C.diagonals i = s(a, b)) ∧
      (∃ i, C.diagonals i = s(b, c)) ∧
      (∃ i, C.diagonals i = s(c, a)) ∧
      -- The triangle has two good diagonals
      ( Diagonal.Good ⟨s(a, b), by simpa using hab.ne⟩ ∧
        Diagonal.Good ⟨s(b, c), by simpa using hbc.ne⟩ ∨
        Diagonal.Good ⟨s(b, c), by simpa using hbc.ne⟩ ∧
        Diagonal.Good ⟨s(c, a), by simpa using (hab.trans hbc).ne'⟩ ∨
        Diagonal.Good ⟨s(c, a), by simpa using (hab.trans hbc).ne'⟩ ∧
        Diagonal.Good ⟨s(a, b), by simpa using hab.ne⟩) ∧
      -- The triangle has two sides of equal length
      ((b.val - a.val : ℤ) = c.val - b.val ∨
       (c.val - b.val : ℤ) = N + a.val - c.val ∨
       (N + a.val - c.val : ℤ) = b.val - a.val)}

abbrev imo_2006_p2_solution : ℕ := sorry

/--
Let $P$ be a regular 2006-gon. A diagonal of $P$ is called good if its endpoints divide the boundary of $P$ into two parts, each composed of an odd number of sides of $P$. The sides of $P$ are also called good. Suppose $P$ has been dissected into triangles by 2003 diagonals, no two of which have a common point in the interior of $P$. Find the maximum number of isosceles triangles having two good sides that could appear in such a configuration.
-/
theorem imo_2006_p2 :
    IsGreatest {k | ∃ c : TriangleDissection 2006, c.numOfIsoscelesTriangle = k}
      imo_2006_p2_solution :=  by sorry
