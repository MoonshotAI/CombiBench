import Mathlib

abbrev Coin := Fin 2

abbrev Coin.H : Coin := 0

abbrev Coin.T : Coin := 1

abbrev CoinConfig (n : ℕ) := Fin n → Coin

def CoinConfig.countH {n : ℕ} (c : CoinConfig n) : ℕ := (List.ofFn c).count .H

def CoinConfig.flip {n : ℕ} (c : CoinConfig n) (k : ℕ) : CoinConfig n :=
  fun i => if i.val + 1 = k then
    match c i with
    | .H => .T
    | .T => .H
  else c i

def CoinConfig.update {n : ℕ} (c : CoinConfig n) : Option (CoinConfig n) :=
  if c.countH = 0 then none else .some <| c.flip c.countH

def CoinConfig.updateMultipleTimes {n : ℕ} (c : CoinConfig n) : ℕ → Option (CoinConfig n)
  | 0 => if c.countH = 0 then none else .some c
  | k+1 => c.updateMultipleTimes k >>= update

abbrev imo_2019_p5_2_solution : ℕ → ℚ := sorry

/--
The Bank of Bath issues coins with an $H$ on one side and a $T$ on the other. Harry has $n$ of these coins arranged in a line from left to right. He repeatedly performs the following operation: If there are exactly $k > 0$ coins showing $H$, then he turns over the $k^{th}$ coin from the left; otherwise, all coins show $T$ and he stops. For example, if $n = 3$ the process starting with the configuration $THT$ would be $THT \rightarrow HHT \rightarrow HTT \rightarrow TTT$, which stops after three operations. (a) Show that, for each initial configuration, Harry stops after a finite number of operations. (b) For each initial configuration $C$, let $L(C)$ be the number of operations before Harry stops. For example, $L(THT) = 3$ and $L(TTT) = 0$. Determine the average value of $L(C)$ over all $2^n$ possible initial configurations $C$.
-/
theorem imo_2019_p5_1 {n : ℕ} (hn : n > 0) : ∀ (c : CoinConfig n), ∃ N : ℕ, c.updateMultipleTimes N = .none := by sorry

theorem imo_2019_p5_2 {n : ℕ} (hn : n > 0) : imo_2019_p5_2_solution n =
    ∑ c : CoinConfig n, (Nat.find (imo_2019_p5_1 hn c) : ℚ) / (Fintype.card (CoinConfig n)) := by sorry
