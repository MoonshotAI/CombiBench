import Mathlib

structure Square where
  (pos : ℝ × ℝ)
  (side_length : ℕ)

noncomputable def Square.vertices (s: Square) : Finset (ℝ × ℝ) :=
  let x := s.pos.1; let y := s.pos.2; let n : ℝ := s.side_length;
  {(x, y), (x + n, y), (x, y + n), (x + n, y + n)}

def Square.occupy (s : Square) : Set (ℝ × ℝ) :=
  let x := s.pos.1; let y := s.pos.2; let n : ℝ := s.side_length;
  {p | x ≤ p.1 ∧ p.1 ≤ x + n ∧ y ≤ p.2 ∧ p.2 ≤ y + n}

def touches_only_vertices (s1: Square) (s2: Square): Prop :=
  ∀ p ∈ s1.occupy ∩ s2.occupy, p ∈ s1.vertices ∧ p ∈ s2.vertices

/--
Let $n \geq 5$ be an integer. Consider $n$ squares with side lengths $1,2, \ldots, n$, respectively. The squares are arranged in the plane with their sides parallel to the $x$ and $y$ axes. Suppose that no two squares touch, except possibly at their vertices.\nShow that it is possible to arrange these squares in a way such that every square touches exactly two other squares.
-/
theorem apmo_2023_p1 (n : ℕ) (h_n: n ≥ 5) :
    ∃ position : Fin n → ℝ × ℝ,
    (∀ n1 n2 : Fin n, n1 ≠ n2 → touches_only_vertices ⟨position n1, n1 + 1⟩ ⟨position n2, n2 + 1⟩) ∧
    ∀ m : Fin n, {S | ∃ i, S = ⟨position i, i + 1⟩ ∧ touches_only_vertices ⟨position m, m + 1⟩ S}.ncard = 2 := by sorry
