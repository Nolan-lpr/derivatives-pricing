# Derivatives Pricing

A progressive implementation of derivatives pricing models in Python, from the Black-Scholes closed form up to a Phoenix autocall structured note


## Modules implemented so far

- `black_scholes.py` : Closed-form European option pricing with its analytical Greeks
- `Monte_Carlo.py` : Monte Carlo pricing of European vanilla options under Black-Scholes dynamics, and comparison to Black-Scholes. And pricing of the Greeks
- `barriere.py` : Monte Carlo pricing of the four standard barrier types on simulated paths, closed-form Reiner-Rubinstein (1991) reference for the down-and-in put, and the Broadie-Glasserman-Kou continuity correction
- `autocall.py` : Monte Carlo pricing of a phoenix autocall note, with memory coupons, European or continuously monitored capital protection, distribution metrics (call probability per date, expected life, loss probability), and delta / vega by finite differences with common random numbers

- Coming (last commit) : Call-back probability by date, controle variable of the autocall, and full explanation and graphs of the product in the jupyter Notebook dedicated to the autocall.


## Repository structure

```
src/         the four pricing modules
tests/       280 unit tests, with AI assistance, one file per module
notebooks/   one analysis notebook per module
```

## Validation

- `test_black_scholes.py` : my Black_Scholes.py pricer is tested against call-put parity, centered finite differences, some monotony tests and classic boundaries.

- `test_monte_carlo.py` : Monte_Carlo.py is tested against Black-Scholes with statistical error, against the martingale identity E(S_T) = S e^((r-q)T), and variance reduction methods are checked to be unbiased and their factor reduction is measured. Validation tests were designed with AI assistance, then debugged and extended by hand. 

- `test_barriere.py` : path simulation is checked against `Monte_Carlo.py` (the terminal column must have the same law whatever the number of steps), prices against in-out parity with common random numbers, against the closed-form reference, and against the degenerate cases where the barrier is unreachable or breached from the start. Validation tests were designed with AI assistance, then debugged and extended by hand. 

- `test_autocall.py` : no closed form exists for a phoenix autocall, so degenerate cases replace the reference: zero coupon with no call is a zero-coupon bond (exact), moving the protection to 100% makes it that bond minus a vanilla put (checked against black_scholes.py), and a zero call level reduces it to a single certain cash flow at the first date. The rest is internal consistency: call probabilities plus survival sum to one, monotonicities have the right sign, and the optional leg is decomposed into a put plus a digital, which gives an exact expected price. 280 tests across the suite. Validation tests were designed with AI assistance, then debugged and extended by hand.

You can try the tests with :
```bash
pytest -v tests/
```

## Requirements 

Numpy, Scipy, Pytest 
Notebooks also need Matplotlib and Pandas


## Performance of the Monte Carlo pricer

European call, S = K = 100, r = 3%, q = 1%, sigma = 20%, T = 1 year, 100 000 paths.

**Convergence.** The standard error decays as N^(-1/2): a log-log fit over
10^3 to 10^6 paths gives a slope of -0.491 against a theoretical -0.5.

**Variance reduction.** Both methods are unbiased; their efficiency depends
entirely on how well the payoff correlates with the control variate S_T.

| Strike | Correlation | Antithetic | Control variate | Theoretical 1/(1-rho²) |
|---|---|---|---|---|
| 50 (deep ITM)  | 0.999997 | 24.9 | 149 463 | 149 463 |
| 100 (ATM)      | 0.9046   | 1.71 | 5.50    | 5.50    |
| 150 (deep OTM) | 0.3651   | 0.99 | 1.15    | 1.15    |

The measured reduction factor matches 1/(1-rho²) exactly, which is expected
since the optimal beta is estimated on the same draws. Note that antithetic
sampling falls below 1 out of the money: on a heavily truncated payoff it
degrades the estimator rather than improving it.

**Greeks and common random numbers.** All five Greeks fall within 1.3 standard
errors of their closed-form values. Reusing the same draws for the bumped and
unbumped prices divides the standard error on delta by a factor of about 17,
so the variance by roughly 285: without it, refining the finite-difference
step h degrades the estimate instead of improving it.

**Black-Scholes PDE check.** Using the Monte Carlo Greeks, the relation
theta + (r-q)·S·delta + ½·sigma²·S²·gamma = r·V holds within 0.15 standard
errors over 20 seeds. The residual is dominated by gamma, whose error is
amplified by a factor sigma²S²/2 = 200 in the equation.


## Discrete monitoring bias

Down-and-in put, S = K = 100, B = 70, r = 3%, q = 1%, sigma = 20%, T = 1 year.
Continuous-monitoring reference: Reiner-Rubinstein closed form.

Monte Carlo on discretely monitored paths systematically **underprices** a
knock-in: a path can cross the barrier between two observation dates and
come back unseen, so crossings are missed and never invented. The bias is
one-directional, and it is a bias rather than noise: increasing the number
of paths tightens the confidence interval around a wrong value.

It decays as m^(-1/2) in the number of monitoring dates (measured slope
-0.417 with 50000 simulations, -0.458 with 200 simulations on a log-log fit), 
which is slow: at 252 steps, that is daily monitoring over a year, 
the bias is still 0,147 which means the common "one step per trading day" 
convention is not enough.

The Broadie-Glasserman-Kou correction shifts the simulated barrier by
exp(+beta·sigma·sqrt(T/m)) with beta = 0.5826, bringing it closer to the
spot so that discrete monitoring reproduces the continuous price. With it,
12 steps already beat 1000 uncorrected steps. The correction overshoots
below roughly 12 steps, where its asymptotic assumption breaks down.

This matters in practice rather than academically: a path matrix costs
8·n_paths·(m+1) bytes, so 100 000 paths over 5000 steps is 4 GB. Reaching by
brute force the accuracy the correction gives at 12 steps is not feasible in
memory.



### The Phoenix autocall

Five-year note, 20 quarterly observation dates, 2% coupon per date (8% a year),
call level 100%, coupon barrier 70%, European protection at 60%, S = 100,
r = 3%, q = 1%, sigma = 20%, 200 000 paths.

| Metric | Value |
|---|---|
| Price | 101.94 +/- 0.03 |
| Expected life | 1.26 years |
| Called at the first date | 49.9% |
| Surviving to maturity | 12.9% |
| Loss probability | 5.7% |
| Average loss given loss | 52.5 per 100 of nominal |

**A five-year maturity with a fifteen-month expected life, and a risk that is
rare but severe.** Half the paths are called at the very first observation, and
each later date only sees the survivors of the previous ones. A 5.7% loss
probability looks reassuring, but by the time the 60% barrier is breached the
underlying has already halved, hence the 52.5 points lost.

**The holder is short volatility**, the opposite of a vanilla option. Measured
vega is -42 per unit of volatility. Lowering the protection barrier to 20%
shrinks it to -9, which confirms the exposure comes from the sold down-and-in
put and not from the coupon leg.

**Delta is not monotone in spot.** Current spot bumped against a fixed
contractual reference level of 100, common random numbers, h = 1%:

| Spot | 50 | 60 | 80 | 95 | 100 | 110 | 140 |
|---|---|---|---|---|---|---|---|
| Delta | 1.68 | 1.39 | 0.49 | 0.04 | -0.03 | -0.04 | 0.00 |

Far below the reference the sold put dominates and the note moves more than the
underlying. The delta crosses zero just below the call level and turns
**negative**: a further rise makes early redemption more likely, which cuts off
the remaining coupons. Far above, the note converges to a near-certain
redemption at the next date, worth
(nominal + one coupon)·exp(-r·T/n_obs) = 101.24.

**Common random numbers are no longer a convenience here.** The payoff is
discontinuous: a path crossing the call level switches a whole block of cash
flows on or off. Reusing the draws across the two bumped legs divides the
variance of the delta by **90**. The log-log slope of the noise against the step
size is -1.00 without them, exactly the 1/h law, and -0.49 with: refining the
step makes the estimate worse in the first case and better in the second.

**Control variate.** A vanilla put struck at the protection barrier, evaluated
on the same paths. It depends only on S_T, so its Black-Scholes expectation is
exact, unlike the closed-form down-and-in put which is a continuous-monitoring
price and would inject the monitoring bias into the result. The variance
reduction is 1.58 at the reference parametrisation and rises to 3.53 at a 150%
call level, matching 1/(1-rho²) throughout. The gain is small where the call is
easy because the control only sees the terminal value, while what differs
between paths there is the **date** at which the note dies, hence the number of
coupons collected.



## Model and numerical limitations

`black_scholes.py` :
We work here with Black-Scholes closed-form, which contains several limitations : 
- constant volatility : this will be a severe issue to price a Phoenix auto call, as the payoff depends on the far left tail of the distribution, exactly where the equity skew is steepest
-  constant dividends and rates : This will not be the case in reality with an auto call
- No jumps, continuous diffusion : BS assumes continuous paths and no jumps/gap, which will not be the case with a barrier option

`Monte_Carlo.py` :
- Statistical error, not model error: the price is an estimate with a standard
  error decaying as N^(-1/2). 
- Do not work with non-constant volatility : sampling S_T in a single step is unbiased because
  the GBM has a closed-form solution. Any local or stochastic volatility model
  will require discretising the path, reintroducing a discretisation bias on top
  of the statistical one.
- Terminal value only: the current simulator returns S_T alone, with no path in
  between. A barrier or an autocall needs the whole trajectory, so this is the
  first thing the next module will extend.

`barriere.py`:
- Discrete monitoring, continuous contract: the simulated path is only
  observed at the n_pas sampling dates, so crossings that happen and revert
  between two dates are missed. The error is one-directional, never the
  reverse, which makes it a bias rather than noise: knock-in options are
  underpriced, knock-out options overpriced. It decays as m^(-1/2), so
  brute force is a poor remedy and the Broadie-Glasserman-Kou correction
  does the real work.
- The correction is asymptotic: valid for small time steps and away from
  the barrier. It overshoots below roughly 12 monitoring dates, and it
  degrades when the spot sits close to the barrier, which is exactly the
  regime where an autocall becomes interesting.
- Memory bound: a path matrix costs 8·n_paths·(m+1) bytes, so 100 000 paths
  over 5000 steps is 4 GB. Since the bias needs many steps and a clean
  measurement of it needs many paths, the two requirements multiply. Only
  the running minimum actually matters for a barrier, so the full path never
  needs to be stored; the current implementation does not exploit this.
- Closed form restricted to B <= K and to the down-and-in put only here

`autocall.py`:
- Scale invariance: every level is a fraction of a reference spot, so a note priced at
  issue does not depend on the spot at all and its delta is identically zero. The
  contractual reference is therefore separated from the current spot (`S_ref`).
- Issue date only: the pricer always starts from a full maturity with every observation
  date ahead. A seasoned note would also need the elapsed dates and the current
  memory-coupon counter.
- Constant volatility hurts more here than anywhere else: the payoff hangs on the far
  left tail, exactly where the equity skew is steepest, and the sold down-and-in put is
  priced off a flat surface.
- The contractual dates carry no monitoring bias, but the continuous protection does,
  and the Broadie-Glasserman-Kou correction is not applied to it yet.
- No control variate: S_T correlates poorly with this payoff, but the closed-form
  down-and-in put is now available and is the natural control for the optional leg.

