# Derivatives Pricing

This is a progressive implementation of derivatives pricing models : We began here by implementing Black-Scholes for vanilla options for now and calculated its associated analytical Greeks.

We will then go step by step up to a Phoenix autocall structured note pricer

## Modules

- `black_scholes.py` : Closed-form European option pricing with its analytical Greeks
- `Monte_Carlo.py` : Monte Carlo pricing of European vanilla options under Black-Scholes dynamics, and comparison to Black-Scholes
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