import Mathlib

open Nat

/--
Prove that there are at least 100! ways to partition the number 100! into summands from the set $\{1!, 2!, 3!, \ldots, 99!\}$. (Partitions differing in the order of summands are considered the same; any summand can be taken multiple times. We remind that $n!=1 \cdot 2 \cdot \ldots \cdot n$.)
-/
theorem izho_2019_p1 : ((@Finset.univ 100!.Partition).filter
    (fun p => ∀ i ∈ p.parts, ∃ k ∈ Finset.Icc 1 99, i = Nat.factorial k)).card ≥ 100! := by sorry
