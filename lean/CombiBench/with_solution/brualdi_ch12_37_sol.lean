import Mathlib

open SimpleGraph BigOperators Classical

variable (n : ℕ) {V : Type*} (G : SimpleGraph V)

def SimpleGraph.IsDominatingSet (D : Set V) : Prop :=
  ∀ v : V, ¬ (v ∈ D) →  ∃ u ∈ D, G.Adj u v

lemma IsDominatingSet.univ : G.IsDominatingSet Set.univ := by simp [IsDominatingSet]

noncomputable def SimpleGraph.eDominationNum : ℕ∞ := iInf (fun s ↦ if
  (G.IsDominatingSet s) then s.card else ⊤ : (Finset V) → ℕ∞)

noncomputable def SimpleGraph.dominationNum : ℕ := G.eDominationNum.toNat

abbrev Q_3 := (pathGraph 2) □ (pathGraph 2) □ (pathGraph 2)


/--
Determine the domination number of the graph $Q_{3}$ of vertices and edges of a three-dimensional cube.
-/
-- TODO surely the domination number is 2 not 1
theorem brualdi_ch12_37 : Q_3.dominationNum = ((1) : ℕ ):= by sorry
