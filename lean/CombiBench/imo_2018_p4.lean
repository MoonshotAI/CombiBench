import Mathlib

abbrev Site := Fin 20 × Fin 20

-- when we see them as real points, their coordinates become within 1,...,20
def Site.asPoint (s : Site) : ℝ × ℝ := (s.1.1 + 1, s.2.1 + 1)

inductive State
  | red
  | blue
  | unoccupied

abbrev Game := Site → State

-- initially the all sites are unoccupied
def initialGame : Game := fun _ => State.unoccupied

def valid_Amy_move (x : Site) (g : Game) : Prop :=
  g x = State.unoccupied ∧
  ∀ y, g y = State.red → dist x.asPoint y.asPoint ≠ √5

def valid_Ben_move (x : Site) (g : Game) : Prop :=
  g x = State.unoccupied

-- Either Amy can still move, or the game is finished
def AmyStrategy := Π (g : Game), Option ((x : Site) ×' valid_Amy_move x g)

-- if Amy can still make a move, then update the game according to her strategy
def Game.updateAccordingToAmyStrategy (g : Game) (s : AmyStrategy) : Option Game :=
  (s g) >>= fun p => .some <| Function.update g p.1 .red

-- Either Ben can still move, or the game is finished
def BenStrategy := Π (g : Game), Option ((x : Site) ×' valid_Ben_move x g)

-- if Ben can still make a move, then update the game according to his strategy
def Game.updateAccordingToBenStrategy (g : Game) (s : BenStrategy) : Option Game :=
  (s g) >>= fun p => .some <| Function.update g p.1 .blue

-- Amy goes first
def updateOneTurn (a : AmyStrategy) (b : BenStrategy) (g : Game) : Option Game :=
  g.updateAccordingToAmyStrategy a >>= fun g' => g'.updateAccordingToBenStrategy b

-- counting from `0`, at the `0`-th turn, the board if empty
-- at the first turn, amy and ben has both placed one stone
-- at the nth turn, amy and ben has both placed n stones
def updateGame (a : AmyStrategy) (b : BenStrategy) (g : Game) : ℕ → Option Game
  | 0 => .some g
  | (n + 1) => updateOneTurn a b g >>= (updateGame a b · n)

def CanPlaceKRedStones (a : AmyStrategy) (b : BenStrategy) : ℕ → Prop
  | 0 => True -- amy can always place at least 0 stones
  | n+1 => -- amy can places (n+1) stones when:
    ∃ (h : updateGame a b initialGame n |>.isSome), -- the game is not over at n-th turn
      a ((updateGame a b initialGame n).get h) |>.isSome -- amy can still make a move at (n+1)-th turn

abbrev imo_2018_p4_solution : ℕ := sorry

/--
A site is any point $(x, y)$ in the plane such that $x$ and $y$ are both positive integers less than or equal to 20. Initially, each of the 400 sites is unoccupied. Amy and Ben take turns placing stones with Amy going first. On her turn, Amy places a new red stone on an unoccupied site such that the distance between any two sites occupied by red stones is not equal to $\sqrt{5}$. On his turn, Ben places a new blue stone on any unoccupied site. (A site occupied by a blue stone is allowed to be at any distance from any other occupied site.) They stop as soon as a player cannot place a stone. Find the greatest $K$ such that Amy can ensure that she places at least $K$ red stones, no matter how Ben places his blue stones.
-/
theorem imo_2018_p4 :
    -- there exists a strategy for Amy, such that no matter how Ben play, Amy can place at least `k` stone.
    (∃ a : AmyStrategy, ∀ b : BenStrategy, CanPlaceKRedStones a b imo_2018_p4_solution) ∧
    -- but no matter how Amy play, there is a strategy for Ben, such that Amy can not place `k+1` stones.
    (∀ a : AmyStrategy, ∃ b : BenStrategy, ¬ CanPlaceKRedStones a b (imo_2018_p4_solution + 1)) := by
    sorry
