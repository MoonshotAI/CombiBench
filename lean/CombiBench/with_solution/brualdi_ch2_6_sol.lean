import Mathlib


/--
How many integers greater than 5400 have both of the following properties? (a) The digits are distinct. (b) The digits 2 and 7 do not occur.
-/
theorem brualdi_ch2_6 (s : Finset ℕ)
    (hs : ∀ n, n ∈ s ↔ n > 5400 ∧ (Nat.digits 10 n).Nodup ∧ 2 ∉ (Nat.digits 10 n) ∧ 7 ∉ (Nat.digits 10 n)) :
    s.card = ((94830) : ℕ ) := by sorry
