import Mathlib

namespace Imo_2024_p5

-- A cell on the board for the game.
abbrev Cell (N : ℕ) : Type := Fin (N + 2) × Fin (N + 1)

-- A row that is neither the first nor the last (and thus contains a monster).
abbrev InteriorRow (N : ℕ) : Type := (Set.Icc 1 ⟨N, by omega⟩ : Set (Fin (N + 2)))

-- Data for valid positions of the monsters.
abbrev MonsterData (N : ℕ) : Type := InteriorRow N ↪ Fin (N + 1)

-- The cells with monsters as a Set, given an injection from rows to columns.
def MonsterData.monsterCells {N} (m : MonsterData N) :
    Set (Cell N) :=
  Set.range (fun x : InteriorRow N ↦ ((x : Fin (N + 2)), m x))

-- Whether two cells are adjacent.
def Adjacent {N} (x y : Cell N) : Prop :=
  Nat.dist x.1 y.1 + Nat.dist x.2 y.2 = 1

-- A valid path from the first to the last row.
structure Path (N : ℕ) where
  -- The cells on the path.
  cells : List (Cell N)
  nonempty : cells ≠ []
  head_first_row : (cells.head nonempty).1 = 0
  last_last_row : (cells.getLast nonempty).1 = N + 1
  valid_move_seq : cells.Chain' Adjacent

-- The first monster on a path, or `none`.
noncomputable def Path.firstMonster {N} (p : Path N) (m : MonsterData N) : Option (Cell N) :=
  letI := Classical.propDecidable
  p.cells.find? (fun x ↦ (x ∈ m.monsterCells : Bool))

-- A strategy, given the results of initial attempts, returns a path for the next attempt.
abbrev Strategy (N : ℕ) : Type := ⦃k : ℕ⦄ → (Fin k → Option (Cell N)) → Path N

-- Playing a strategy, k attempts.
noncomputable def Strategy.play {N} (s : Strategy N) (m : MonsterData N) :
    (k : ℕ) → Fin k → Option (Cell N)
  | 0 => Fin.elim0
  | k + 1 => Fin.snoc (s.play m k) ((s (s.play m k)).firstMonster m)

-- The predicate for a strategy winning within the given number of attempts.
def Strategy.WinsIn {N} (s : Strategy N) (m : MonsterData N) (k : ℕ) : Prop :=
  none ∈ Set.range (s.play m k)

-- Whether a strategy forces a win within the given number of attempts.
def Strategy.ForcesWinIn {N} (s : Strategy N) (k : ℕ) : Prop :=
  ∀ m, s.WinsIn m k

abbrev imo_2024_p5_solution : ℕ := sorry

/--
Turbo the snail plays a game on a board with 2024 rows and 2023 columns. There are hidden monsters in 2022 of the cells. Initially, Turbo does not know where any of the monsters are, but he knows that there is exactly one monster in each row except the first row and the last row, and that each column contains at most one monster. Turbo makes a series of attempts to go from the first row to the last row. On each attempt, he chooses to start on any cell in the first row, then repeatedly moves to an adjacent cell sharing a common side. (He is allowed to return to a previously visited cell.) If he reaches a cell with a monster, his attempt ends and he is transported back to the first row to start a new attempt. The monsters do not move, and Turbo remembers whether or not each cell he has visited contains a monster. If he reaches any cell in the last row, his attempt ends and the game is over. Determine the minimum value of $n$ for which Turbo has a strategy that guarantees reaching the last row on the $n^{th}$ attempt or earlier, regardless of the locations of the monsters.
-/
theorem imo_2024_p5 : IsLeast {k | ∃ s : Strategy 2022, s.ForcesWinIn k} imo_2024_p5_solution := by
    sorry
