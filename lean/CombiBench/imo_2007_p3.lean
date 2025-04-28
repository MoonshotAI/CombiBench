import Mathlib

/--
In a mathematical competition some competitors are friends. Friendship is always mutual. Call a group of competitors a clique if each two of them are friends. (In particular, any group of fewer than two competitors is a clique.) The number of members of a clique is called its size. Given that, in this competition, the largest size of a clique is even, prove that the competitors can be arranged in two rooms such that the largest size of a clique contained in one room is the same as the largest size of a clique contained in the other room.
-/
theorem imo_2007_p3 {player : Type} [Fintype player] (math_competiton : SimpleGraph player)
    (h : Even math_competiton.cliqueNum) :
    ∃ a : SimpleGraph.Subgraph math_competiton, a.coe.cliqueNum = aᶜ.coe.cliqueNum := by sorry
