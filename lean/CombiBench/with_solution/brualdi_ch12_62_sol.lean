import Mathlib

structure TwoConnected {V : Type*} (G : SimpleGraph V) : Prop where
  selfconnected : G.Connected
  remains_connected : ∀ x : V, ((⊤ : SimpleGraph.Subgraph G).deleteVerts {x}).coe.Connected

/--
Let $G$ be a graph. Prove that $G$ is 2-connected if and only if, for each vertex $x$ and each edge $\alpha$, there is a cycle that contains both the vertex $x$ and the edge $\alpha$.
-/
theorem brualdi_ch12_62 {V : Type*} (G : SimpleGraph V) : TwoConnected G ↔ ∀ x : V, ∀ e ∈ G.edgeSet,
    ∃ G' : SimpleGraph.Subgraph G, x ∈ G'.verts ∧ e ∈ G'.edgeSet ∧ G'.coe.IsCycles := by sorry
