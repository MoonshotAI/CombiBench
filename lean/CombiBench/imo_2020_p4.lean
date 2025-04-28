import Mathlib

def Iscompanies (n k : ℕ) (car_set : Finset (Fin n × Fin n)) : Prop :=
  car_set.card = k ∧
  (∀ (a b : Fin n), (a, b) ∈ car_set → a < b)∧
  (∀ a ∈ car_set, ∀ b ∈ car_set, a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2)

def Islinked {n : ℕ} (a b : Fin n) (car_set : Finset (Fin n × Fin n)) : Prop :=
  ∃ s : List (Fin n × Fin n), s.Nodup ∧ (∀ i ∈ s, (i ∈ car_set ∧
  (List.foldl (fun x y => if x.2 = y.1 then (x.1, y.2) else x) (a, a) s = (a, b))))

def Condition (n k : ℕ) : Prop :=
  ∃ (companyA companyB : Finset (Fin n × Fin n)), Iscompanies n k companyA ∧ Iscompanies n k companyB ∧
  (∃ (a b : Fin n), a ≠ b ∧ Islinked a b companyA ∧ Islinked a b companyB)

abbrev imo_2020_p4_solution : ℕ → ℕ := sorry

/--
There is an integer $n > 1$. There are $n^2$ stations on a slope of a mountain, all at different altitudes. Each of two cable car companies, $A$ and $B$, operates $k$ cable cars; each cable car provides a transfer from one of the stations to a higher one (with no intermediate stops). The $k$ cable cars of $A$ have $k$ different starting points and $k$ different finishing points, and a cable car that starts higher also finishes higher. The same conditions hold for $B$. We say that two stations are linked by a company if one can start from the lower station and reach the higher one by using one or more cars of that company (no other movements between stations are allowed). Determine the smallest positive integer k for which one can guarantee that there are two stations that are linked by both companies.
-/
theorem imo_2020_p4 (n : ℕ) (hn : n > 1) : IsLeast {k | Condition n k} (imo_2020_p4_solution n) := by sorry
