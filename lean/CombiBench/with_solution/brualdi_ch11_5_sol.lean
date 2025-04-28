import Mathlib

/--
Use the pigeonhole principle to prove that a graph of order n ≥ 2 always has two vertices of the same degree.
-/
theorem brualdi_ch11_5 {V : Type*} (n : ℕ) (h_n: n ≥ 2) (G : SimpleGraph (Fin n)) [DecidableRel G.Adj] :
    ∃ v1 v2, v1 ≠ v2 ∧ G.degree v1 = G.degree v2 := by sorry
