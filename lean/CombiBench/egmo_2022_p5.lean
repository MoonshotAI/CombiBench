import Mathlib

def formsDomino (n k : ℕ) (i j : Fin n × Fin (2 * k)) : Bool :=
  -- i and j are on the same row and (i is left to j or j is left to i)
  (i.1.val = j.1.val ∧ (i.2.val + 1 = j.2.val ∨ j.2.val + 1 = i.2.val)) ∨ -- or
  -- i and j are on the same column and (i is above j or j is above i)
  (i.2.val = j.2.val ∧ (i.1.val + 1 = j.1.val ∨ j.1.val + 1 = i.1.val))

structure PerfectCover (n k : ℕ) where
  -- the collections of tiles
  d : Fin (n * k) → ((Fin n × Fin (2 * k)) × (Fin n × Fin (2 * k)))
  -- each tile is a domino
  domino : ∀ i, formsDomino n k (d i).1 (d i).2
  -- every position on the board is covered by some dominos
  covers : ∀ i : Fin n × Fin (2 * k), ∃ j, i = (d j).1 ∨ i = (d j).2

noncomputable instance {n k} : Fintype (PerfectCover n k) :=
  Fintype.ofInjective PerfectCover.d <| by
    rintro ⟨d, _⟩ ⟨d', _⟩ (rfl : d = d')
    rfl

abbrev egmo_2022_p5_solution : Set ℕ := sorry

/--
For all positive integers $n, k$, let $f(n, 2k)$ be the number of ways an $n \times 2k$ board can be fully covered by $nk$ dominoes of size $2 \times 1$. (For example, $f(2,2)=2$ and $f(3,2)=3$.)\nFind all positive integers $n$ such that for every positive integer $k$, the number $f(n, 2k)$ is odd.
-/
theorem egmo_2022_p5 : {n | n > 0 ∧ ∀ k > 0, Odd (Fintype.card (PerfectCover n k))} =
    egmo_2022_p5_solution := by sorry
