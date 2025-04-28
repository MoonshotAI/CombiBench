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

abbrev brualdi_ch12_37_solution : ℕ := sorry

/--
Determine the domination number of the graph $Q_{3}$ of vertices and edges of a three-dimensional cube.
-/
theorem brualdi_ch12_37 : Q_3.dominationNum = brualdi_ch12_37_solution:= by sorry
