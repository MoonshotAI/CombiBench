import Mathlib

def move (k : Fin 2021) (order : Fin 2021 ≃ Fin 2021) : Fin 2021 ≃ Fin 2021 :=
  -- order.symm k is the number of the hole of the k-th walnut
  order.trans (Equiv.swap (order (finRotate _ (order.symm k))) (order ((finRotate _).symm (order.symm k))))

def performMoves (originalOrder : Fin 2021 ≃ Fin 2021) : (Fin 2021) → (Fin 2021 ≃ Fin 2021)
  | 0 => originalOrder
  | ⟨n + 1, lt⟩ => move n (performMoves originalOrder ⟨n, lt_trans (by omega) lt⟩)

/--
Two squirrels, Bushy and Jumpy, have collected 2021 walnuts for the winter. Jumpy numbers the walnuts from 1 through 2021, and digs 2021 little holes in a circular pattern in the ground around their favourite tree. The next morning Jumpy notices that Bushy had placed one walnut into each hole, but had paid no attention to the numbering. Unhappy, Jumpy decides to reorder the walnuts by performing a sequence of 2021 moves. In the $k$-th move, Jumpy swaps the positions of the two walnuts adjacent to walnut $k$. Prove that there exists a value of $k$ such that, on the $k$-th move, Jumpy swaps some walnuts $a$ and $b$ such that $a < k < b$.
-/
theorem imo_2021_p5 (originalOrder : Fin 2021 ≃ Fin 2021) :
    ∃ k, min (finRotate _ ((performMoves originalOrder k).symm k) : ℕ)
        ((finRotate _).symm ((performMoves originalOrder k).symm k) : ℕ) < (k : ℕ) ∧
      (k : ℕ) < max (finRotate _ ((performMoves originalOrder k).symm k) : ℕ)
        ((finRotate _).symm ((performMoves originalOrder k).symm k) : ℕ) := by sorry
