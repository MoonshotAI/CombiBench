import Mathlib

open Equiv Finset

/--
An integer $N \ge 2$ is given. A collection of $N(N + 1)$ soccer players, no two of whom are of the same height, stand in a row. Sir Alex wants to remove $N(N - 1)$ players from this row leaving a new row of $2N$ players in which the following $N$ conditions hold: ($1$) no one stands between the two tallest players, ($2$) no one stands between the third and fourth tallest players, $\cdots$ ($N$) no one stands between the two shortest players. Show that this is always possible.
-/
theorem imo_2017_p5 (N : ℕ) (h_N : N ≥ 2) (height : Perm (Fin (N * (N + 1)))) :
    ∃ kept : Fin (2 * N) ↪o Fin (N * (N + 1)),
    -- For any i, j, such that the ith kept player in the line has an even number kept players shorter than them
    ∀ i j, Even #{l | height (kept l) < height (kept i)} →
      -- and the jth kept player has one more kept player shorter than them
      #{l | height (kept l) < height (kept i)} + 1 = #{l | height (kept l) < height (kept j)} →
        -- There is no kept player between the ith and jth kept players
         (¬ ∃ k, (i < k ∧ k < j) ∨ (j < k ∧ k < i)) := by sorry
