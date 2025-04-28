import Mathlib

abbrev Cards := Fin 100

abbrev Boxes := Fin 3

abbrev Allocation := { f : Cards → Boxes | Function.Surjective f }

-- The trick is that given the sum of any two cards, the magician can tell a box number
abbrev Trick := ℕ → Boxes

-- The trick works when:
def trick_works (f : Allocation) (t : Trick) : Prop :=
  ∀ c₁ c₂ : Cards,
  -- given the sum of two cards from box 0 and box 1 then the trick gives the result of box 2 **and**
  (f.1 c₁ = 0 → f.1 c₂ = 1 → t (c₁.1 + c₂.1) = 2) ∧
  -- given the sum of two cards from box 0 and box 2 then the trick gives the result of box 1 **and**
  (f.1 c₁ = 0 → f.1 c₂ = 2 → t (c₁.1 + c₂.1) = 1) ∧
  -- given the sum of two cards from box 1 and box 2 then the trick gives the result of box 0
  (f.1 c₁ = 1 → f.1 c₂ = 2 → t (c₁.1 + c₂.1) = 0)

abbrev imo_2000_p4_solution : ℕ := sorry

/--
A magician has one hundred cards numbered $1$ to $100$. He puts them into three boxes, a red one, a white one and a blue one, so that each box contains at least one card. A member of the audience selects two of the three boxes, chooses one card from each and announces the sum of the numbers on the chosen cards. Given this sum, the magician identifies the box from which no card has been chosen. How many ways are there to put all the cards into the boxes so that this trick always works? (Two ways are considered different if at least one card is put into a different box.)
-/
theorem imo_2000_p4 (good_allocations : Finset Allocation)
    (h : ∀ f ∈ good_allocations, ∃ (t : Trick), trick_works f t) :
    good_allocations.card = imo_2000_p4_solution := by sorry
