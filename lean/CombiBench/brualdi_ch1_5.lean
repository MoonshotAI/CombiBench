import Mathlib

abbrev Board := Fin 3 × Fin 4

def formsDomino (i j : Board) : Bool :=
  -- i and j are on the same row and (i is left to j or j is left to i)
  (i.1.val = j.1.val ∧ (i.2.val + 1 = j.2.val ∨ j.2.val + 1 = i.2.val)) ∨ -- or
  -- i and j are on the same column and (i is above j or j is above i)
  (i.2.val = j.2.val ∧ (i.1.val + 1 = j.1.val ∨ j.1.val + 1 = i.1.val))

structure PerfectCover where
  -- the collections of tiles
  tiles : Fin 6 → (Board × Board)
  -- each tile is a domino
  domino : ∀ i, formsDomino (tiles i).1 (tiles i).2
  -- every position on the board is covered by some dominos
  covers : ∀ i : Board, ∃ j, i = (tiles j).1 ∨ i = (tiles j).2

noncomputable instance : Fintype PerfectCover :=
  Fintype.ofInjective PerfectCover.tiles <| by
    rintro ⟨tiles, _⟩ ⟨tiles', _⟩ (rfl : tiles = tiles')
    rfl

abbrev brualdi_ch1_5_solution : ℕ := sorry

/--
Find the number of different perfect covers of a 3-by-4 chessboard by dominoes.
-/
theorem brualdi_ch1_5 : Fintype.card PerfectCover = brualdi_ch1_5_solution := by sorry
