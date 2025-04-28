import Mathlib

def weight_of_left_pan {n : ℕ} (σ : Equiv.Perm (Fin n)) (place_left : Fin n → Bool) (step : ℕ) : ℕ :=
  ∑ j with place_left j = true ∧ σ j ≤ step, 2 ^ j.1

def weight_of_right_pan {n : ℕ} (σ : Equiv.Perm (Fin n)) (place_left : Fin n → Bool) (step : ℕ) : ℕ :=
  ∑ j with place_left j = false ∧ σ j ≤ step, 2 ^ j.1

def is_valid_placement {n : ℕ} (σ : Equiv.Perm (Fin n)) (place_left : Fin n → Bool) : Prop :=
  ∀ step : Fin n, weight_of_right_pan σ place_left step ≤ weight_of_left_pan σ place_left step

instance {n : ℕ} (σ : Equiv.Perm (Fin n)) (place_left : Fin n → Bool) :
  Decidable (is_valid_placement σ place_left) := by
  simp [is_valid_placement]; infer_instance

abbrev all_placements (n : ℕ) :=
  Finset.product (@Finset.univ (Equiv.Perm (Fin n)) _) (@Finset.univ (Fin n → Bool) _)

abbrev valid_placements (n : ℕ) :=
  all_placements n |>.filter (fun (σ, f) => is_valid_placement σ f)

abbrev imo_2011_p4_solution : ℕ → ℕ := sorry

/--
Let $n > 0$ be an integer. We are given a balance and $n$ weights of weight $2^0,2^1, \cdots ,2^{n-1}$. We are to place each of the $n$ weights on the balance, one after another, in such a way that the right pan is never heavier than the left pan. At each step we choose one of the weights that has not yet been placed on the balance, and place it on either the left pan or the right pan, until all of the weights have been placed. Determine the number of ways in which this can be done.
-/
theorem imo_2011_p4 (n : ℕ) (hn : n > 0) :
    (valid_placements n).card = imo_2011_p4_solution n := by sorry
