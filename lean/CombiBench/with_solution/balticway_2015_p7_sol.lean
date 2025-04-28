import Mathlib

open SimpleGraph Finset

abbrev Ladies := Fin 100

/--
There are 100 members in a ladies' club. Each lady has had tea (in private) with exactly 56 of the other members of the club. The Board, consisting of the 50 most distinguished ladies, have all had tea with one another. Prove that the entire club may be split into two groups in such a way that, within each group, any lady has had tea with any other.
-/
theorem balticway_2015_p7 (had_tea: SimpleGraph (Ladies)) [DecidableRel had_tea.Adj]
    (h_had_tea_with_56: ∀ l : Ladies, had_tea.degree l = 56)
    (h_board: ∃ board : Finset Ladies, board.card = 50 ∧ (∀ l1 l2: board, had_tea.Adj l1 l2)) :
    ∃ group1 group2: Finset Ladies,
      group1 ∪ group2 = Finset.univ
      ∧ Disjoint group1 group2
      ∧ (∀ l1 l2: group1, had_tea.Adj l1 l2)
      ∧ (∀ l1 l2: group2, had_tea.Adj l1 l2) := by sorry
