import Mathlib

structure PreNecklaces where
  c : Fin 7 → Fin 2
  deriving Fintype

def myDihedralGroup (n : ℕ) : Subgroup (Equiv.Perm (Fin n)) :=
  Subgroup.closure {finRotate n, Fin.revPerm}

instance Necklaces.setoid : Setoid PreNecklaces where
  r n1 n2 := ∃ s ∈ myDihedralGroup 7, n1.c = n2.c ∘ s
  iseqv :=
  { refl n := ⟨1, one_mem _, by simp⟩
    symm := by
      rintro m n ⟨p, hp, eqp⟩
      refine ⟨p⁻¹, inv_mem hp, eqp ▸ ?_⟩
      ext x
      simp
    trans := by
      rintro a b c ⟨p, hp, eqp⟩ ⟨q, hq, eqq⟩
      refine ⟨q * p, mul_mem hq hp, ?_⟩
      rw [eqp, eqq]
      ext x
      simp }

abbrev Necklaces := Quotient Necklaces.setoid

noncomputable instance : Fintype Necklaces := by
  have := Quotient.finite (Necklaces.setoid)
  exact Fintype.ofFinite Necklaces

abbrev brualdi_ch14_26_solution : ℕ := sorry

/--
How many different necklaces are there that contain four red and three blue beads?
-/
theorem brualdi_ch14_26 : Fintype.card Necklaces = brualdi_ch14_26_solution := by sorry
