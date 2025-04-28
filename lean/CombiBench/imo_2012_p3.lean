import Mathlib

open scoped Finset

set_option autoImplicit false

-- A strategy for Alice in the liar guessing game consists of:
structure AliceStrategy where
  -- A natural number `N`.
  N : ℕ
  -- A natural number less than `N`.
  x : Fin N
  -- For each history, a question, in the form of a set of natural numbers.
  nextAnswer : List (Set (Fin N) × Bool) → Set (Fin N) → Bool

-- A strategy for Bob in the liar guessing game consists of:
structure BobStrategy where
  -- For each history, a question, in the form of a set of natural numbers.
  nextQuestion N : List (Set (Fin N) × Bool) → Set (Fin N)
  -- For each history, a guess as to what `x` is, in the form of a small set of natural numbers.
  guess N : List (Set (Fin N) × Bool) → Finset (Fin N)

variable {k n : ℕ}

-- The history of the game is the list of question-answer pairs asked by Bob and answered by Alice.
def history (A : AliceStrategy) (B : BobStrategy) : ℕ → List (Set (Fin A.N) × Bool)
  | 0 => []
  | t + 1 =>
    (B.nextQuestion A.N (history A B t),
      A.nextAnswer (history A B t) (B.nextQuestion A.N (history A B t)))
        :: history A B t

-- A strategy for Alice is valid if every interval of `k` consecutive answers contains some true answer.
def AliceStrategy.IsValid (A : AliceStrategy) (B : BobStrategy) (k : ℕ) : Prop :=
  ∀ t₀ : ℕ, ∃ t ∈ Finset.Ico t₀ (t₀ + k),
    A.nextAnswer (history A B t) (B.nextQuestion A.N (history A B t))
      = (A.x ∈ B.nextQuestion A.N (history A B t))

-- A strategy for Bob is valid at time `t` if the guessing set is of size at most `n`.
def BobStrategy.IsValid (A : AliceStrategy) (B : BobStrategy) (n t : ℕ) : Prop :=
  #(B.guess A.N (history A B t)) ≤ n

-- A strategy for Bob is winning if, for all valid strategies for Alice, there is some time `t` at which the strategy for Bob is valid and `x` belongs in the guessing set.
def BobStrategy.IsWinning (B : BobStrategy) (k n : ℕ) : Prop :=
  ∀ (A : AliceStrategy), A.IsValid B k → ∃ t, B.IsValid A n t ∧ A.x ∈ B.guess A.N (history A B t)

/--
The liar’s guessing game is a game played between two players A and B. The rules of the game depend on two positive integers $k$ and $n$ which are known to both players. At the start of the game the player A chooses integers $x$ and $N$ with $1 \le x \le N$. Player A keeps $x$ secret, and truthfully tells $N$ to the player B. The player B now tries to obtain information about $x$ by asking player A questions as follows: each question consists of B specifying an arbitrary set $S$ of positive integers (possibly one specified in some previous question), and asking A whether $x$ belongs to $S$. Player B may ask as many questions as he wishes. After each question, player A must immediately answer it with yes or no, but is allowed to lie as many times as she wants; the only restriction is that, among any $k+1$ consecutive answers, at least one answer must be truthful. After B has asked as many questions as he wants, he must specify a set $X$ of at most $n$ positive integers. If $x \in X$, then B wins; otherwise, he loses. Prove that: (a) If $n \ge 2^k$ then B has a winning strategy. (b) There exists a positive integer $k_0$ such that for every $k \ge k_0$ there exists an integer $n \ge 1.99^k$ for which B cannot guarantee a victory.
-/
theorem imo_2012_p3 :
    -- If `2 ^ k ≤ n`, then there exists a winning strategy for Bob.
    (∀ k n, 2 ^ k ≤ n → ∃ B : BobStrategy, B.IsWinning k n) ∧
    -- There exists a positive integer `k₀` such that for every `k ≥ k₀` there exists an integer
    -- `n ≥ 1.99 ^ k` such that no strategy for Bob is winning.
      ∃ k₀,
        ∀ k ≥ k₀,
          ∃ n : ℕ, n ≥ (1.99 : ℝ) ^ k ∧
            ∀ B : BobStrategy, ¬ B.IsWinning k n := by sorry
