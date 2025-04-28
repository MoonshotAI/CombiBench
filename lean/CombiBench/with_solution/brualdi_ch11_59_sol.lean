import Mathlib

open SimpleGraph

/--
Prove that the removal of an edge from a tree leaves a forest of two trees.
-/
theorem brualdi_ch11_59 {V : Type*} [Fintype V] [DecidableEq V]
    (T : SimpleGraph V) (hT : IsTree T) (e : Sym2 V) (he : e ∈ T.edgeSet) :
    ∃ (T1 T2 : SimpleGraph V), IsTree T1 ∧ IsTree T2 ∧
    T1.edgeSet ∪ T2.edgeSet = T.edgeSet \ {e} ∧
    Disjoint (T1.support) (T2.support) := by sorry
