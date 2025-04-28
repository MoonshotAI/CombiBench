import Mathlib

def solved_by_at_least_three {n : ℕ} (problem : ℕ) (solved_problems: Fin n → Finset ℕ) : Prop :=
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ problem ∈ solved_problems a ∩ solved_problems b ∩ solved_problems c

/--
Twenty-one girls and twenty-one boys took part in a mathematical competition. It turned out that each contestant solved at most six problems, and for each pair of a girl and a boy, there was at least one problem that was solved by both the girl and the boy. Show that there is a problem that was solved by at least three girls and at least three boys.
-/
theorem imo_2001_p3 (solved_problems_girls: Fin 21 → Finset ℕ)
    (solved_problems_boys: Fin 21 → Finset ℕ)
    (h_max_6_girls: ∀ girl: Fin 21, (solved_problems_girls girl).card ≤ 6)
    (h_max_6_boys: ∀ boy: Fin 21, (solved_problems_boys boy).card ≤ 6)
    (h_pairs: ∀ boy girl: Fin 21, solved_problems_boys boy ∩ solved_problems_girls girl ≠ ∅):
    ∃ problem: ℕ, solved_by_at_least_three problem solved_problems_girls ∧
    solved_by_at_least_three problem solved_problems_boys := by sorry
