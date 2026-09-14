# Derivatives Pricing

This is a progressive implementation project of derivatives : I began here by implementing Black-Scholes for vanilla option for now and calculated its associated analytical Greeks.

We will then go step by step up to a Phoenix autocall structured note pricer

## Modules

- black_scholes.py : Closed-form European option pricing with its analytical Greeks
- Coming : comparison to Monte-Carlo vanilla pricer, Down-and-in barrier option, phoenix auto call pricer...

## Validation

- test_black_scholes.py : my Black_Scholes.py pricer is tested against call-put parity, centered finite method, some monotony tests and classic boundaries using the following command : 
```bash
pytest -v tests/
```

## Requirements 

Numpy, Scipy, Pytest

## Limitations of the model


We work here with Black-Scholes close-form, which contains several limitations : 
- constant volatility : this will be a severe issue to price a Phoenix auto call, as the payoff of the put down-and-in option sold by the investor depends of a very low strike, when the smile is the higher
-  constant dividends and rates : This will not be the case in reality with an auto call
- No jumps, continuous diffusion : BS assumes continuous paths and no jumps/gap, which will not be the case with a barrier option

