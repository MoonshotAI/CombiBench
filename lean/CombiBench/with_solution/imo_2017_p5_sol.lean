import Mathlib

open Equiv Finset

/--
An integer $N \ge 2$ is given. A collection of $N(N + 1)$ soccer players, no two of whom are of the same height, stand in a row. Sir Alex wants to remove $N(N - 1)$ players from this row leaving a new row of $2N$ players in which the following $N$ conditions hold: ($1$) no one stands between the two tallest players, ($2$) no one stands between the third and fourth tallest players, $\cdots$ ($N$) no one stands between the two shortest players. Show that this is always possible.
-/
theorem imo_2017_p5 (N : ℕ) (h_N : N ≥ 2) (height : Perm (Fin (N * (N + 1)))) :
    ∃ kept : Fin (2 * N) ↪o Fin (N * (N + 1)),
    ∀ i j, Even #{l | height (kept l) < height (kept i)} →
      (∀ k, height (kept i) < height (kept k) ↔ height (kept j) ≤ height (kept k)) →
      (∀ k, kept i < kept k ↔ kept j ≤ kept k) ∨ (∀ k, kept j < kept k ↔ kept i ≤ kept k) := by sorry
