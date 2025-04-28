import Mathlib

open List Lex

/--
The complement $\bar{A}$ of an $r$-subset $A$ of $\{1,2, \ldots, n\}$ is the $(n-r)$-subset of $\{1,2, \ldots, n\}$, consisting of all those elements that do not belong to $A$. Let $M=\binom{n}{r}$, the number of $r$-subsets and, at the same time, the number of $(n-r)$ subsets of $\{1,2, \ldots, n\}$. Prove that, if $A_{1}, A_{2}, A_{3}, \ldots, A_{M}$ are the $r$-subsets in lexicographic order, then $\overline{A_{M}}, \ldots, \overline{A_{3}}, \overline{A_{2}}, \overline{A_{1}}$ are the $(n-r)$-subsets in lexicographic order.
-/
theorem brualdi_ch4_35 (r n M : ℕ) (hM : M = ((@Finset.univ (Fin n)).powersetCard r).card)
    (A : Fin M → (Finset.powersetCard r (@Finset.univ (Fin M) _))) :
    ∀ i j, (List.Lex (fun x1 x2 : Fin M => x1 ≤ x2)
    (Finset.sort (· ≤ ·) (A i)) (Finset.sort (· ≤ ·) (A j))) →
    (List.Lex (fun x1 x2 : Fin M => x1 ≤ x2)
    (Finset.sort (· ≤ ·) (A j)ᶜ) (Finset.sort (· ≤ ·) (A i)ᶜ)) := by sorry
