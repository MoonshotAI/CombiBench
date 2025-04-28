import Mathlib

def S := Finset.Icc 1 1000000

/--
$S$ is the set $\{1, 2, 3, \dots ,1000000\}$. Show that for any subset $A$ of $S$ with $101$ elements we can find $100$ distinct elements $x_i$ of $S$, such that the sets $\{a + x_i \mid a \in A\}$ are all pairwise disjoint.
-/
theorem imo_2003_p1 (A : Finset S) (hA: A.card = 101):
    ∃ x : Function.Embedding (Fin 100) S,
    ∀ i j, i ≠ j → Disjoint { a.1 + (x i).1 | a ∈ A } { a.1 + (x j).1 | a ∈ A } := by sorry
