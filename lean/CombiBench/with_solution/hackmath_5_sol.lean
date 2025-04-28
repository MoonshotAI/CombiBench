import Mathlib

open MeasureTheory ProbabilityTheory ENNReal


/--
There are 10 red marbles, 6 green marbles, and 4 blue marbles in a box. What is the probability of picking the next red marble?
-/
theorem hackmath_5 : uniformOn (Ω := (Fin 20)) ⊤ {i | i.1 < 10} = ((1 / 2) : ENNReal ) := by sorry
