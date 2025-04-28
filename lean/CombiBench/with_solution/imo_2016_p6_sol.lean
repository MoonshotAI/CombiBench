import Mathlib

open Function

-- A `N`-frog system is the data of `N * (N - 1) / 2` points and `N` segments straight in the plane of each `N - 1` points such that every two segments intersect in exactly one point.
structure FrogSystem (N : ℕ) where
  -- For each segment `s` and time `t`, the `t`-th segment `s'` that `s` meets.
  otherSegment (s : Fin N) : Fin (N - 1) ≃ {s' : Fin N // s ≠ s'}
  -- The point indexed by the pair of segments it belongs to.
  point : {p : Sym2 (Fin N) // ¬ p.IsDiag} → EuclideanSpace ℝ (Fin 2)
  -- Points are in order along each segment.
  mem_collinear {s t₀ t₁ t₂} : t₀ < t₁ → t₁ < t₂ → Sbtw ℝ
      (point ⟨s(s, otherSegment s t₀), by simpa using (otherSegment s t₀).2⟩)
      (point ⟨s(s, otherSegment s t₁), by simpa using (otherSegment s t₁).2⟩)
      (point ⟨s(s, otherSegment s t₂), by simpa using (otherSegment s t₂).2⟩)

-- Geoff's wish is that for each time `t` the `t` intersections of each segment are different.
def FrogSystem.GeoffsWish {N : ℕ} (F : FrogSystem N) : Prop :=
  ∀ t, Injective fun s ↦ s(s, F.otherSegment s t)

/--
There are $n\ge 2$ line segments in the plane such that every two segments cross and no three segments meet at a point. Geoff has to choose an endpoint of each segment and place a frog on it facing the other endpoint. Then he will clap his hands $n-1$ times. Every time he claps, each frog will immediately jump forward to the next intersection point on its segment. Frogs never change the direction of their jumps. Geoff wishes to place the frogs in such a way that no two of them will ever occupy the same intersection point at the same time. (a) Prove that Geoff can always fulfill his wish if $n$ is odd. (b) Prove that Geoff can never fulfill his wish if $n$ is even.
-/
theorem imo_2016_p6 :
    -- If `n ≥ 2` is odd, then Geoff can always fulfill his wish.
    (∀ n ≥ 2, Odd n → ∃ F : FrogSystem n, F.GeoffsWish) ∧
    -- If `n ≥ 2` is even, then Geoff can never fulfill his wish.
      ∀ n ≥ 2, Even n → ∀ F : FrogSystem n, ¬ F.GeoffsWish := by sorry
