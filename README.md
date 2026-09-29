# Derivatives Pricing

This is a progressive implementation of derivatives pricing models : We began here by implementing Black-Scholes for vanilla options for now and calculated its associated analytical Greeks.

We will then go step by step up to a Phoenix autocall structured note pricer

## Modules

- `black_scholes.py` : Closed-form European option pricing with its analytical Greeks
- `Monte_Carlo.py` : Monte Carlo pricing of European vanilla options under Black-Scholes dynamics, and comparison to Black-Scholes. And pricing of the Greeks
- Coming : Down-and-in barrier option, phoenix auto call pricer...

## Validation

- `test_black_scholes.py` : my Black_Scholes.py pricer is tested against call-put parity, centered finite differences, some monotony tests and classic boundaries.

- `test_monte_carlo.py` : Monte_Carlo.py is tested against Black-Scholes with statistical error, against the martingale identity E(S_T) = S e^((r-q)T), and variance reduction methods are checked to be unbiased and their factor reduction is measured. Validation tests were designed with AI assistance, then debugged and extended by hand. 

You can try the tests with :
```bash
pytest -v tests/
```

## Requirements 

Numpy, Scipy, Pytest 

### Performance of the Monte Carlo pricer

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