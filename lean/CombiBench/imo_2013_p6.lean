import Mathlib

open Equiv Fintype Finset

def IsBeautiful {n} (e : Perm (Fin (n + 1))) : Prop :=
  ∀ ⦃a b⦄, a < b → ∀ ⦃c⦄, b < c → ∀ ⦃d⦄, c < d → a.val + d.val = b.val + c.val →
    -- a bc d
    e a < e b ∧ e b < e d ∧ e a < e c ∧ e c < e d ∨
    -- d bc a
    e d < e b ∧ e b < e a ∧ e d < e c ∧ e c < e a ∨
    -- b ad c
    e b < e a ∧ e a < e c ∧ e b < e d ∧ e d < e c ∨
    -- c ad b
    e c < e a ∧ e a < e b ∧ e c < e d ∧ e d < e b ∨
    -- ad bc
    e a < e b ∧ e a < e c ∧ e d < e b ∧ e d < e c ∨
    -- bc ad
    e b < e a ∧ e c < e a ∧ e b < e d ∧ e c < e d

instance {n} : DecidablePred (IsBeautiful (n := n)) := by unfold IsBeautiful; infer_instance

def M (n : ℕ) : ℕ := #{e : Perm (Fin (n + 1)) | IsBeautiful e} / (n + 1)

def N (n : ℕ) : ℕ := #{xy ∈ .Icc 1 n ×ˢ .Icc 1 n | xy.1 + xy.2 ≤ n ∧ xy.1.gcd xy.2 = 1}

/--
Let $n \ge 3$ be an integer, and consider a circle with $n + 1$ equally spaced points marked on it. Consider all labellings of these points with the numbers $0, 1, ... , n$ such that each label is used exactly once; two such labellings are considered to be the same if one can be obtained from the other by a rotation of the circle. A labelling is called beautiful if, for any four labels $a < b < c < d$ with $a + d = b + c$, the chord joining the points labelled $a$ and $d$ does not intersect the chord joining the points labelled $b$ and $c$. Let $M$ be the number of beautiful labelings, and let N be the number of ordered pairs $(x, y)$ of positive integers such that $x + y \le n$ and $\gcd(x, y) = 1$. Prove that\[ M = N + 1.\]
-/
theorem imo_2013_p6 (n : ℕ) (hn : n ≥ 3) : M n = N n + 1 := by sorry
