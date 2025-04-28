import Mathlib

open Nat

def f {m : ℕ} (n : Finset.Icc 1 m → ℤ) (x : Equiv.Perm (Finset.Icc 1 m)) : ℤ := ∑ i, x i * n i

/--
Let $n_1, n_2, \dots , n_m$ be integers where $m>1$ is odd. Let $x = (x_1, \dots , x_m)$ denote a permutation of the integers $1, 2, \cdots , m$. Let $f(x) = x_1n_1 + x_2n_2 + ... + x_mn_m$. Show that for some distinct permutations $a$, $b$ the difference $f(a) - f(b)$ is a multiple of $m!$.
-/
theorem imo_2001_p4 (m : ℕ) (h_m_pos: m > 1) (h_m_odd: Odd m) (n : Finset.Icc 1 m → ℤ):
    ∃ a b : Equiv.Perm (Finset.Icc 1 m), a ≠ b ∧ ↑(m !) ∣ (f n a - f n b) := by sorry
