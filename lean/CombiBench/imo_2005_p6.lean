import Mathlib

/--
In a mathematical competition, in which 6 problems were posed to the participants, every two of these problems were solved by more than 2/5 of the contestants. Moreover, no contestant solved all the 6 problems. Show that there are at least 2 contestants who solved exactly 5 problems each.
-/
theorem imo_2005_p6 {participants : Type} [Fintype participants] [DecidableEq participants]
    (solved : Fin 6 → Finset participants)
    (h : ∀ i j, i ≠ j → (solved i ∩ solved j).card > (2 * Fintype.card participants : ℝ) / 5)
    (h' : ∀ i, (solved i).card < Fintype.card participants) :
    ∃ s : Finset participants, s.card ≥ 2 ∧
    (∀ i ∈ s, ∃ p : Finset (Fin 6), p.card = 5 ∧ ∀ j, j ∈ p ↔ i ∈ solved j) := by sorry
