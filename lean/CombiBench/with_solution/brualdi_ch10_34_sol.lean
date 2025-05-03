import Mathlib

structure SteinerTripleSystemOfIndOne (t k n : ℕ) where
  carrier : Fin n
  blocks : Finset (Finset (Fin n))
  card_blocks : ∀ b ∈ blocks, b.card = k
  block_inner : ∀ s : (Finset (Fin n)), s.card = t → ∃! b ∈ blocks, s ⊆ b

/--
Let $t$ be a positive integer. Prove that, if there exists a Steiner triple system of index 1 having $v$ varieties, then there exists a Steiner triple system having $v^{t}$ varieties.
-/
-- TODO Why t > 1 rather than t ≥ 1?
-- TODO Also it looks to me like only the first steiner triple system is required to have index 1, not the second one.
-- Also, in the definition above, I think n is what the text calls `v` so why not use that notation?
theorem brualdi_ch10_34 (t v : ℕ) (ht : t > 1): Nonempty (SteinerTripleSystemOfIndOne 2 3 v) →
    Nonempty (SteinerTripleSystemOfIndOne 2 3 (v ^ t)) := by sorry
