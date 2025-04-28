import Mathlib

abbrev brualdi_ch2_6_solution : ℕ := sorry

/--
How many integers greater than 5400 have both of the following properties? (a) The digits are distinct. (b) The digits 2 and 7 do not occur.
-/
theorem brualdi_ch2_6 (s : Finset ℕ)
    (hs0 : ∀ n ∈ s, n > 5400)
    (hs1 : ∀ n ∈ s, (Nat.digits 10 n).Nodup)
    (hs2 : ∀ n ∈ s, 2 ∉ (Nat.digits 10 n) ∧ 7 ∉ (Nat.digits 10 n)) :
    s.card = brualdi_ch2_6_solution := by sorry
