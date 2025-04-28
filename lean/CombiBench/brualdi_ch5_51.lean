import Mathlib

/--
Let $R$ and $S$ be two partial orders on the same set $X$. Considering $R$ and $S$ as subsets of $X \times X$, we assume that $R \subseteq S$ but $R \neq S$. Show that there exists an ordered pair $(p, q)$, where $(p, q) \in S$ and $(p, q) \notin R$ such that $R^{\prime}=R \cup\{(p, q)\}$ is also a partial order on $X$.
-/
theorem brualdi_ch5_51 {X : Type} [DecidableEq X] (R S : Rel X X) [IsPartialOrder X R]
    [IsPartialOrder X S] (le : R < S) :
    ∃ (p q : X), S p q ∧ ¬ R p q ∧
    IsPartialOrder X (R ⊔ fun x y ↦ if x = p ∧ y = q then true else false) := by sorry
