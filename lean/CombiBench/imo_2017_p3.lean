import Mathlib

abbrev imo_2017_p3_solution : Bool := sorry

/--
A hunter and an invisible rabbit play a game in the Euclidean plane. The rabbit's starting point, $A_0$, and the hunter's starting point, $B_0$, are the same. After $n-1$ rounds of the game, the rabbit is at point $A_{n-1}$ and the hunter is at point $B_{n-1}$. In the nth round of the game, three things occur in order. (i) The rabbit moves invisibly to a point $A_n$ such that the distance between $A_{n-1}$ and $A_n$ is exactly 1. (ii) A tracking device reports a point $P_n$ to the hunter. The only guarantee provided by the tracking device is that the distance between $P_n$ and $A_n$ is at most 1. (iii) The hunter moves visibly to a point $B_n$ such that the distance between $B_{n-1}$ and $B_n$ is exactly 1. Is it always possible, no matter how the rabbit moves, and no matter what points are reported by the tracking device, for the hunter to choose her moves so that after $10^9$ rounds she can ensure that the distance between her and the rabbit is at most 100?
-/
theorem imo_2017_p3 (start : ℝ × ℝ) : imo_2017_p3_solution =
    ∀ (A : ℕ → (Fin 2 → ℝ)), A 0 = ![start.1, start.2] → ∀ n, dist (A n) (A (n + 1)) = 1 →
    (∃ (P : ℕ → (Fin 2 → ℝ)), ∀ n > 0, dist (P n) (A n) ≤ 1) →
    (∃ (B : ℕ → (Fin 2 → ℝ)), B 0 = ![start.1, start.2] ∧ ∀ n, dist (B n) (B (n + 1)) = 1 ∧
    dist (A (10 ^ 9)) (B (10 ^9)) ≤ 100) := by sorry
