import Mathlib

structure SteinerTripleSystemOfIndOne (t k n : ℕ) where
  carrier : Fin n
  blocks : Finset (Finset (Fin n))
  card_blocks : ∀ b ∈ blocks, b.card = k
  block_inner : ∀ s : (Finset (Fin n)), s.card = t → ∃! b ∈ blocks, s ⊆ b

structure SteinerTripleSystem (t k n I : ℕ) where
  carrier : Fin n
  blocks : Finset (Finset (Fin n))
  card_blocks : ∀ b ∈ blocks, b.card = k
  block_inner : ∀ s : (Finset (Fin n)), s.card = t → ∃ f : Fin I ↪ blocks, ∀ i, s ⊆ f i

/--
Let $t$ be a positive integer. Prove that, if there exists a Steiner triple system of index 1 having $v$ varieties, then there exists a Steiner triple system having $v^{t}$ varieties.
-/
theorem brualdi_ch10_34 (t v : ℕ) (ht : t > 0) : Nonempty (SteinerTripleSystemOfIndOne 2 3 v) →
    ∃ I, Nonempty (SteinerTripleSystem 2 3 (v ^ t) I) := by sorry
