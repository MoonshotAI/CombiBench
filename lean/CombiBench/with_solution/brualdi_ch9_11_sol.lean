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

/--
Let $n>1$, and let $\mathcal{A}=\left(A_{1}, A_{2}, \ldots, A_{n}\right)$ be the family of subsets of $\{1,2, \ldots, n\}$, where \[ A_{i}=\{1,2, \ldots, n\}-\{i\}, \quad(i=1,2, \ldots, n) \] Prove that $\mathcal{A}$ has an SDR and that the number of SDRs is the $n$th derangement number $D_{n}$.
-/
theorem brualdi_ch9_11 (n : ℕ) (hn : n > 1) (A : Fin n → Finset ℕ)
    (hA : ∀ i, A i = Finset.Icc 1 n \ {i.1 + 1}) :
    Nonempty (SDR A) ∧ Fintype.card (SDR A) = numDerangements n := by sorry
