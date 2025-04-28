import Mathlib

def balanced (S : Set (ℝ × ℝ)) : Prop :=
  ∀ A ∈ S, ∀ B ∈ S, A ≠ B → (∃ C ∈ S, dist A C = dist B C)

def centre_free (S : Set (ℝ × ℝ)) : Prop :=
  ∀ A ∈ S, ∀ B ∈ S, ∀ C ∈ S, A ≠ B → B ≠ C → A ≠ C →
    ¬ (∃ P ∈ S, dist A P = dist B P ∧ dist B P = dist C P)

abbrev imo_2015_p1_solution : Set ℕ := sorry

/--
We say that a finite set $\mathcal{S}$ in the plane is balanced if, for any two different points $A$, $B$ in $\mathcal{S}$, there is a point $C$ in $\mathcal{S}$ such that $AC=BC$. We say that $\mathcal{S}$ is centre-free if for any three points $A$, $B$, $C$ in $\mathcal{S}$, there is no point $P$ in $\mathcal{S}$ such that $PA=PB=PC$. (1) Show that for all integers $n\geq 3$, there exists a balanced set consisting of $n$ points. (1) Determine all integers $n\geq 3$ for which there exists a balanced centre-free set consisting of $n$ points.
-/
theorem imo_2015_p1 : (∀ n ≥ 3, ∃ (S : Finset (ℝ × ℝ)), balanced S ∧ S.card = n) ∧
    {n | n ≥ 3 ∧ (∃ (S : Finset (ℝ × ℝ)), balanced S ∧ centre_free S ∧ S.card = n)} =
    imo_2015_p1_solution := by sorry
