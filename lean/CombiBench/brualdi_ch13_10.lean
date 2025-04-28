import Mathlib

universe u

structure IsTournament {V : Type u} (G : Digraph V) : Prop where
  irrefl : ∀ (u : V), ¬ G.Adj u u
  adj : ∀ (u v : V), u ≠ v → (G.Adj u v ↔ ¬ G.Adj v u)

-- a walk is a sequence of edges such that the end of the `i`-th edge is the start of the `i+1`-th edge
inductive Digraph.Walk {V : Type u} (G : Digraph V) : V → V → Type u
  | nil {u : V} (h : G.Adj u u) : Digraph.Walk G u u
  | cons {u v w : V} (h : G.Adj u v) (p : Digraph.Walk G v w) : Digraph.Walk G u w
  deriving DecidableEq

-- the support of a walk is the list of vertices that the walk traverses in order.
def Digraph.Walk.support {V : Type u} {G : Digraph V} {u v : V} : Digraph.Walk G u v → List V
  | .nil h => [u]
  | .cons _ p => u :: p.support

-- a walk is a path if it does not visit the same vertex twice, hence a Hamilton cycle is a path
-- that starts and ends at the same vertex.
def Digraph.Walk.IsPath {V : Type u} {G : Digraph V} {u v : V} (p : Digraph.Walk G u v) : Prop :=
  p.support.Nodup

def Digraph.Walk.length {V : Type u} {G : Digraph V} {u v : V} : Digraph.Walk G u v → ℕ
  | .nil h => 0
  | .cons _ p => 1 + p.length

/--
Prove that every tournament contains a vertex $u$ such that, for every other vertex $x$, there is a path from $u$ to $x$ of length at most 2.
-/
theorem brualdi_ch13_10 {V : Type u} (T : Digraph V) (hT : IsTournament T) :
    ∃ (u : V), ∀ (x : V), ∃ (p : T.Walk u x), p.IsPath ∧ p.length ≤ 2 := by sorry
