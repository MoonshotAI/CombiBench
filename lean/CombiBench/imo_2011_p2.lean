import Mathlib

open EuclideanGeometry Real Filter

namespace List

variable {α β : Type*}

-- This is already in a later version of mathlib than the one we are depending on
-- Whether a predicate holds for all ordered triples of elements of a list.
@[mk_iff]
inductive Triplewise (p : α → α → α → Prop) : List α → Prop
  | nil : [].Triplewise p
  | cons {a : α} {l : List α} : l.Pairwise (p a) → l.Triplewise p → (a :: l).Triplewise p

end List

instance : Fact (Module.finrank ℝ (EuclideanSpace ℝ (Fin 2)) = 2) := ⟨by simp⟩

variable {n} [Fintype n] [DecidableEq n]

-- The value of the following doesn't matter. All that matters is that it exists
noncomputable instance : Module.Oriented ℝ (EuclideanSpace ℝ n) n where
  positiveOrientation := Basis.orientation (Pi.basisFun ℝ n)

structure IsWindmillProcess (S : Set (EuclideanSpace ℝ (Fin 2)))
    (f : ℕ → EuclideanSpace ℝ (Fin 2)) where
  forall_mem n : f n ∈ S
  oangle_le_oangle n x : x ∈ S →
    toIocMod two_pi_pos 0 (oangle (f n) (f (n + 1)) (f (n + 2))).toReal
      ≤ toIocMod two_pi_pos 0 (oangle (f n) (f (n + 1)) x).toReal

/--
Let $\mathcal{S}$ be a finite set of at least two points in the plane. Assume that no three points of $\mathcal S$ are collinear. A windmill is a process that starts with a line $\ell$ going through a single point $P \in \mathcal S$. The line rotates clockwise about the pivot $P$ until the first time that the line meets some other point belonging to $\mathcal S$. This point, $Q$, takes over as the new pivot, and the line now rotates clockwise about $Q$, until it next meets a point of $\mathcal S$. This process continues indefinitely. Show that we can choose a point $P$ in $\mathcal S$ and a line $\ell$ going through $P$ such that the resulting windmill uses each point of $\mathcal S$ as a pivot infinitely many times.
-/
theorem imo_2011_p2 (l : List (EuclideanSpace ℝ (Fin 2)))
    (hl : l.Triplewise (¬ Collinear ℝ {·, ·, ·})) :
    ∃ f : ℕ → EuclideanSpace ℝ (Fin 2),
      IsWindmillProcess {x | x ∈ l} f ∧ ∀ x ∈ l, ∃ᶠ n in atTop, f n = x := by sorry
