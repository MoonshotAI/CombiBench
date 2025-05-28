import Mathlib

abbrev Cards := Finset.Icc 1 100

abbrev Boxes := Fin 3

-- The trick is that given the sum of any two cards, the magician can tell a box number
abbrev Trick := ℕ → Boxes

-- The trick works when:
def trick_works (f : Cards → Boxes) (t : Trick) : Prop :=
  ∀ c₁ c₂ : Cards,
  -- given the sum of two cards from box 0 and box 1 then the trick gives the result of box 2 **and**
  (f c₁ = 0 → f c₂ = 1 → t (c₁.1 + c₂.1) = 2) ∧
  -- given the sum of two cards from box 0 and box 2 then the trick gives the result of box 1 **and**
  (f c₁ = 0 → f c₂ = 2 → t (c₁.1 + c₂.1) = 1) ∧
  -- given the sum of two cards from box 1 and box 2 then the trick gives the result of box 0
  (f c₁ = 1 → f c₂ = 2 → t (c₁.1 + c₂.1) = 0)


/--
A magician has one hundred cards numbered $1$ to $100$. He puts them into three boxes, a red one, a white one and a blue one, so that each box contains at least one card. A member of the audience selects two of the three boxes, chooses one card from each and announces the sum of the numbers on the chosen cards. Given this sum, the magician identifies the box from which no card has been chosen. How many ways are there to put all the cards into the boxes so that this trick always works? (Two ways are considered different if at least one card is put into a different box.)
-/
theorem imo_2000_p4 (good_allocations : Finset (Cards → Boxes))
    (h : ∀ f, f ∈ good_allocations ↔ Function.Surjective f ∧ ∃ (t : Trick), trick_works f t) :
    good_allocations.card = ((12) : ℕ ) := by sorry
