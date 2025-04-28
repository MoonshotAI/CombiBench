import Mathlib

/--
Let $m$ and $n$ be positive integers whose greatest common divisor is $d$. Prove that the greatest common divisor of the Fibonacci numbers $f_{m}$ and $f_{n}$ is the Fibonacci number $f_{d}$.
-/
theorem brualdi_ch7_7 (m n d : ℕ+) (hmd : d = Nat.gcd m n) :
    Nat.gcd (Nat.fib m) (Nat.fib n) = Nat.fib d := by sorry
