import Mathlib

/--
Prove that the complement of a disconnected graph is connected.
-/
theorem brualdi_ch12_34 {V : Type*} (G : SimpleGraph V) (h : ¬ G.Connected) :
    Gᶜ.Connected := by sorry
