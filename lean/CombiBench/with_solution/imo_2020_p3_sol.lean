import Mathlib

structure Pebble (n : ℕ) where
  weight : ℕ
  color : Fin n
deriving DecidableEq

/--
There are $4n$ pebbles of weights $1, 2, 3, \ldots, 4n$. Each pebble is colored in one of $n$ colors and there are four pebbles of each color. Show that we can arrange the pebbles into two piles so that the following two conditions are both satisfied: 1) The total weights of both piles are the same. 2) Each pile contains two pebbles of each color.
-/
theorem imo_2020_p3 (n : ℕ) (PebbleSet : Finset (Pebble n)) (hP : PebbleSet.card = 4 * n)
    (h_weight : ∀ p ∈ PebbleSet, p.weight ∈ Finset.Icc 1 (4 * n))
    (h_ne_weight : ∀ p ∈ PebbleSet, ∀ q ∈ PebbleSet, p ≠ q → p.weight ≠ q.weight)
    (h_color : ∀ i, (PebbleSet.filter (fun p => p.color = i)).card = 4) :
    ∃ (P1 P2 : Finset (Pebble n)), P1 ∪ P2 = PebbleSet ∧ P1 ∩ P2 = ∅ ∧
    ∑ p ∈ P1, p.weight = ∑ p ∈ P2, p.weight ∧ (∀ i, (P1.filter (fun p => p.color = i)).card = 2) ∧
    (∀ i, (P2.filter (fun p => p.color = i)).card = 2) := by sorry
