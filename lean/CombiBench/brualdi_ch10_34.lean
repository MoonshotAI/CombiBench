import Mathlib

structure SteinerTripleSystemOfIndOne (t k n : ℕ) where
  carrier : Fin n
  blocks : Finset (Finset (Fin n))
  card_blocks : ∀ b ∈ blocks, b.card = k
  block_inner : ∀ s : (Finset (Fin n)), s.card = t → ∃! b ∈ blocks, s ⊆ b

/--
Let $t$ be a positive integer. Prove that, if there exists a Steiner triple system of index 1 having $v$ varieties, then there exists a Steiner triple system having $v^{t}$ varieties.
-/
theorem brualdi_ch10_34 (t v : ℕ) (ht : t > 1): Nonempty (SteinerTripleSystemOfIndOne 2 3 v) →
    Nonempty (SteinerTripleSystemOfIndOne 2 3 (v ^ t)) := by sorry
