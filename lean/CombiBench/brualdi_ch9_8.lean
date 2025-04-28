import Mathlib

variable {α : Type*}

structure SDR {n : ℕ} (A : Fin n → Finset α) where
  toFun : Fin n → α
  mem_Ai : ∀ i, toFun i ∈ A i
  pairwise : ∀ i j, i ≠ j → toFun i ≠ toFun j

instance {n : ℕ} (A : Fin n → Finset α) : CoeFun (SDR A) (fun _ => Fin n → α) where
  coe s := s.toFun

noncomputable instance {n : ℕ} (A : Fin n → Finset α) : Fintype (SDR A) := by
  classical
  let Y := Finset.biUnion (@Finset.univ (Fin n) _) A
  if h : Nonempty (SDR A) then
    exact Fintype.ofSurjective (α := (Fin n → Y))
      (fun f ↦ if h1 : (∃(g : SDR A), ∀ i, f i = g i) then ⟨fun i => f i,
          fun i ↦ by have ⟨g, hg⟩ := h1; simp [hg, g.mem_Ai],
          fun i j hij ↦ by have ⟨g, hg⟩ := h1; simp [hg, g.pairwise i j hij]⟩
        else Classical.choice (α := (SDR A)) h) <| fun g ↦
          ⟨fun i => ⟨g i, by simp [Y]; use i; simp [g.mem_Ai]⟩, by
            simp; suffices ∃ (g' : SDR A), ∀ (i : Fin n), g.toFun i = g'.toFun i by simp [this]
            use g; simp⟩
  else exact fintypeOfNotInfinite (fun h1 ↦ by aesop)

abbrev A : Fin 6 → Finset ℕ := fun i ↦ match i with
  | 1 => {1, 2}
  | 2 => {2, 3}
  | 3 => {3, 4}
  | 4 => {4, 5}
  | 5 => {5, 6}
  | 6 => {6, 1}

abbrev brualdi_ch9_8_solution : ℕ := sorry

/--
Let $\mathcal{A}=\left(A_{1}, A_{2}, A_{3}, A_{4}, A_{5}, A_{6}\right)$, where \[ \begin{aligned} & A_{1}=\{1,2\}, A_{2}=\{2,3\}, A_{3}=\{3,4\} \\ & A_{4}=\{4,5\}, A_{5}=\{5,6\}, A_{6}=\{6,1\} \end{aligned} \] Determine the number of different SDRs that $\mathcal{A}$ has.
-/
theorem brualdi_ch9_8 : Fintype.card (SDR A) = brualdi_ch9_8_solution := by sorry
