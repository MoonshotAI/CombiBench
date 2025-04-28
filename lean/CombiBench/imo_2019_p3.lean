import Mathlib

noncomputable instance (users : SimpleGraph (Fin 2019)) (a : Fin 2019) : Fintype (users.neighborSet a) :=
  Fintype.ofFinite ↑(users.neighborSet a)

def IsTriple (l : List (Fin 2019)) (G : SimpleGraph (Fin 2019)) : Prop :=
  l.length = 3 ∧ G.Adj l[0]! l[1]! ∧ G.Adj l[0]! l[2]! ∧ ¬ G.Adj l[1]! l[2]! ∧ l[1]! ≠ l[2]!

@[simp]
lemma triple.hab' (l : List (Fin 2019)) (users : SimpleGraph (Fin 2019)) (h : IsTriple l users) :
  l[1]! = l[2]! ↔ False := by
  constructor
  · exact (h.2.2.2.2 ·)
  · tauto

attribute [local aesop safe] SimpleGraph.Adj.symm IsTriple

-- For any such triple of users a, b, c, b and c becomes friends now, but a is no longer friends with b, and no longer friends with c.
def update (l : List (Fin 2019)) (users : SimpleGraph (Fin 2019)) (h : IsTriple l users) :
  SimpleGraph (Fin 2019) where
    Adj x y :=
      if x = l[0]! then
        if y = l[1]! then False
        else if y = l[2]! then False
        else users.Adj l[0]! y
      else if x = l[1]! then
        if y = l[0]! then False
        else if y = l[2]! then True
        else users.Adj l[1]! y
      else if x = l[2]! then
        if y = l[0]! then False
        else if y = l[1]! then True
        else users.Adj l[2]! y
      else users.Adj x y
    symm := by
      simp only [if_false_left, if_true_left]
      intro x y
      simp only
      split_ifs <;> aesop
    loopless := by
      intro x
      simp only [if_false_left, if_true_left, SimpleGraph.irrefl, if_false_right]
      split_ifs with h1 h2
      · aesop
      · subst h2
        simp_all only [not_false_eq_true, SimpleGraph.irrefl, imp_false, Decidable.not_not, true_and]
        apply Aesop.BuiltinRules.not_intro
        intro a_1
        apply h.2.2.2.2; simp [a_1]
      · aesop

structure ExpectSeq where
  l : List (SimpleGraph (Fin 2019) × List (Fin 2019))
  h_length : l.length > 0
  h_card : ∀ li ∈ l, li.2.length = 3
  h_triple : ∀ li ∈ l, IsTriple li.2 li.1
  h_update : ∀ i : Fin l.length, (finRotate _ i).1 ≠ 0 →
    update l[i].2 l[i].1 (h_triple l[i] (by simp)) = l[finRotate _ i].1

def final_state (seq : ExpectSeq) : SimpleGraph (Fin 2019) :=
  update seq.l[((finRotate seq.l.length).symm ⟨0, by simp [seq.h_length]⟩ )]!.2
    seq.l[((finRotate seq.l.length).symm ⟨0, by simp [seq.h_length]⟩)]!.1
    (seq.h_triple seq.l[((finRotate seq.l.length).symm ⟨0, by simp [seq.h_length]⟩)]! (by simp))

/--
A social network has $2019$ users, some pairs of whom are friends. Whenever user $A$ is friends with user $B$, user $B$ is also friends with user $A$. Events of the following kind may happen repeatedly, one at a time: Three users $A$, $B$, and $C$ such that $A$ is friends with both $B$ and $C$, but $B$ and $C$ are not friends, change their friendship statuses such that $B$ and $C$ are now friends, but $A$ is no longer friends with $B$, and no longer friends with $C$. All other friendship statuses are unchanged. Initially, $1010$ users have $1009$ friends each, and $1009$ users have $1010$ friends each. Prove that there exists a sequence of such events after which each user is friends with at most one other user.
-/
theorem imo_2019_p3 (users : SimpleGraph (Fin 2019))
    (cond : ∃ (A B : Finset (Fin 2019)), A.card = 1010 ∧ B.card = 1009 ∧
    (∀ a ∈ A, (users.neighborFinset a).card = 1009) ∧
    (∀ b ∈ B, (users.neighborFinset b).card = 1010)) :
    ∃ seq : ExpectSeq, ∀ i, ((final_state seq).neighborFinset i).card ≤ 1 := by sorry
