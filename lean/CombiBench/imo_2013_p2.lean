import Mathlib

structure Colombian_config : Type where
  (R B : Finset (ℝ × ℝ))
  (hR : R.card = 2013)
  (hB : B.card = 2014)
  (hC : R ∩ B = ∅)
  (h_no_collinear : ∀ p ∈ R ∪ B, ∀ q ∈ R ∪ B, ∀ r ∈ R ∪ B, p ≠ q → p ≠ r → q ≠ r →
    ¬ ∃ t : ℝ, t ≠ 0 ∧ t * (q.1 - p.1) = (r.1 - p.1) ∧ t * (q.2 - p.2) = (r.2 - p.2))

def Good_arrange (C : Colombian_config) (L : Finset (ℝ × ℝ × ℝ)) : Prop :=
  (∀ l ∈ L, l.1 ≠ 0 ∨ l.2.1 ≠ 0) ∧
  (∀ p ∈ C.R ∪ C.B, ∀ l ∈ L, l.1 * p.1 + l.2.1 * p.2 + l.2.2 ≠ 0) ∧
    ¬ (∃ q ∈ C.R, ∃ p ∈ C.B, ∀ l ∈ L,
      Real.sign (l.1 * p.1 + l.2.1 * p.2 + l.2.2) = Real.sign (l.1 * q.1 + l.2.1 * q.2 + l.2.2))

abbrev imo_2013_p2_solution : ℕ := sorry

/--
A configuration of $4027$ points in the plane is called Colombian if it consists of $2013$ red points and $2014$ blue points, and no three of the points of the configuration are collinear. By drawing some lines, the plane is divided into several regions. An arrangement of lines is good for a Colombian configuration if the following two conditions are satisfied: (1) no line passes through any point of the configuration; (2)no region contains points of both colours. Find the least value of $k$ such that for any Colombian configuration of $4027$ points, there is a good arrangement of $k$ lines.
-/
theorem imo_2013_p2 : IsLeast
    {k | ∀ C : Colombian_config, ∃ L : Finset (ℝ × ℝ × ℝ), L.card = k ∧ Good_arrange C L}
    imo_2013_p2_solution := by sorry
