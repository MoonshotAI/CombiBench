import Mathlib

abbrev sortedList (n : ℕ) := (List.range (2 * n))|>.map
  (fun i ↦ if i < n then 0 else 1)

def checkList (k : ℕ) : List ℕ → ℕ × ℕ := fun L ↦ Id.run do
  let mut i0 := k - 1
  let mut i1 := k - 1
  for i in [k : L.length] do
    if L[i]! = L[k-1]! then
      i1 := i1 + 1
    else break
  for j in [1 : k] do
    if L[k-1-j]! = L[k-1]! then
      i0 := i0 - 1
    else break
  return (i0, i1)

abbrev action (k : ℕ) : List ℕ → List ℕ := fun L ↦
  (List.range ((checkList k L).2 - (checkList k L).1 + 1)).map (fun _ ↦ L[k-1]!) ++
  (List.range (checkList k L).1).map (fun i ↦ L[i]!) ++
  (List.range (L.length - (checkList k L).2 - 1)).map (fun i ↦ L[i + (checkList k L).2 + 1]!)

abbrev pown (k m : ℕ) : List ℕ → List ℕ := fun L ↦ Id.run do
  let mut L' := L
  for _ in [0 : m] do
    L' := action k L'
  return L'

abbrev checkLeft (n : ℕ) : List ℕ → Bool := fun L ↦ Id.run do
  for i in [0 : n] do
    if L[i]! ≠ L[0]! then
      return false
    else continue
  return true

def initial (n : ℕ) : Finset (List ℕ) := (List.replicate n 0 ++ List.replicate n 1).permutations.toFinset

abbrev imo_2022_p1_solution : ℕ → Set (ℕ × ℕ) := sorry

 /--
 The Bank of Oslo issues two types of coin: aluminium (denoted A) and bronze (denoted B). Marianne has $n$ aluminium coins and $n$ bronze coins, arranged in a row in some arbitrary initial order. A chain is any subsequence of consecutive coins of the same type. Given a fixed positive integer $k\le 2n$, Marianne repeatedly performs the following operation: she identifies the longest chain containing the $k^{th}$ coin from the left, and moves all coins in that chain to the left end of the row. For example, if $n = 4$ and $k = 4$, the process starting from the ordering AABBBABA would be $AABBBABA \rightarrow BBBAAABA \rightarrow AAABBBBA \rightarrow BBBBAAAA \rightarrow BBBBAAAA \rightarrow \ldots$. Find all pairs $(n, k)$ with $1 \le k \le 2n$ such that for every initial ordering, at some moment during the process, the leftmost $n$ coins will all be of the same type.
 -/
 theorem imo_2022_p1 (n : ℕ) (hn : n > 0) :
    {(n', k) | n = n' ∧ k ≥ 1 ∧ k ≤ 2 * n ∧ (∀ I ∈ initial n, ∃ m : ℕ, checkLeft n' (pown k m I))} =
    imo_2022_p1_solution n := by sorry
