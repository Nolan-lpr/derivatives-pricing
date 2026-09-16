"""
We price here a vanilla option from the Black-Scholes formula, and compute its Greeks: Delta, Gamma, Vega, Theta, Rho
This is the first part of a bigger project: pricing a Phoenix autocall structured note

"""

import numpy as np
from scipy.stats import norm


# ------------------------------------------------------
#                 Definition d1 et d2
# ------------------------------------------------------
def _d1_d2(S, K, r, q, sigma, T):
    #'regulier' permet juste d'être sûr qu'on travail avec quelque chose de cohérent (sigma * sqrt(T) > 0)

    S, K, r, q, sigma, T = (np.asarray(x, dtype=float)
                            for x in (S, K, r, q, sigma, T))
    S, K, r, q, sigma, T = np.broadcast_arrays(S, K, r, q, sigma, T)

    vol_sqrt_t = sigma * np.sqrt(T)
    regulier = vol_sqrt_t > 0
    denom = np.where(regulier, vol_sqrt_t, 1.0)

    dans_la_monnaie_forward = S * np.exp(-q * T) > K * np.exp(-r * T)
    d1 = np.where(
        regulier,
        (np.log(np.where(S > 0, S, 1.0) / K) + (r - q + 0.5 * sigma ** 2) * T) / denom,
        np.where(dans_la_monnaie_forward, np.inf, -np.inf),
    )
    d2 = d1 - np.where(regulier, vol_sqrt_t, 0.0)
    return d1, d2, regulier
    

def _scalaire_si_possible(x): 
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x


# ------------------------------------------------------
#                  PRICER call et put
# ------------------------------------------------------
def price(S, K, r, q, sigma, T, kind="call"):
    d1, d2, regulier = _d1_d2(S, K, r, q, sigma, T)
    S, K, r, q, T = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (S, K, r, q, T)))

    spot_actualise = S * np.exp(-q * T)
    strike_actualise = K * np.exp(-r * T)

    if kind == "call":
        valeur = spot_actualise * norm.cdf(d1) - strike_actualise * norm.cdf(d2)
        payoff_degenere = np.maximum(spot_actualise - strike_actualise, 0.0)
    elif kind == "put":
        valeur = strike_actualise * norm.cdf(-d2) - spot_actualise * norm.cdf(-d1)
        payoff_degenere= np.maximum(strike_actualise - spot_actualise,0.0)
    else:
        raise ValueError("kind doit valoir 'call' ou 'put'")

    return _scalaire_si_possible(np.where(regulier, valeur, payoff_degenere))

#Let's verificate the call-put parity just to be sure
def verif_parite_call_put(S,K,r,q,sigma,T):
    call = price(S, K, r, q, sigma, T, kind="call")
    put = price(S, K, r, q, sigma, T, kind="put")
    ecart = abs((call - put) - (S*np.exp(-q*T) - K*np.exp(-r*T)))

    if ecart<1e-8:
        print(f"la parité call-put est respécté ! : (écart= {ecart : .2e})")
    else :
        raise ValueError(f"problème de parité call-put ! (écart = {ecart: .2e})")


# ------------------------------------------------------
#                      GREEKS
# ------------------------------------------------------
def delta(S, K, r, q, sigma, T, kind="call"):
    d1, d2, regulier = _d1_d2(S,K,r,q,sigma,T)
    if kind == "call":
        valeur = np.exp(-q*T)*norm.cdf(d1)
    elif kind == "put":
        valeur = -np.exp(-q*T)*norm.cdf(-d1)
    else:
            raise ValueError("Name Error'")
    return _scalaire_si_possible(valeur)


def gamma(S, K, r, q, sigma, T,kind="call"):  # gamma est le même pour un put aussi
    d1,d2,regulier = _d1_d2(S,K,r,q,sigma,T)
    valeur = ( np.exp(-q*T)*norm.pdf(d1) ) / (S*sigma*np.sqrt(T))
    return _scalaire_si_possible(np.where(regulier,valeur,0.0))


def vega(S, K, r, q, sigma, T, kind="call"): # même remarque : vega est le même pour un put aussi
    d1,d2,regulier= _d1_d2(S,K,r,q,sigma,T)
    valeur = S*norm.pdf(d1)*np.exp(-q*T)*np.sqrt(T)  # ici on est par unité de vol (pas point de vol)
    return _scalaire_si_possible(np.where(regulier,valeur,0.0))



def rho(S, K, r, q, sigma, T, kind="call"): 
    d1,d2,regulier = _d1_d2(S,K,r,q,sigma,T)
    if kind == "call":
        valeur = K*T*norm.cdf(d2)*np.exp(-r*T)
    elif kind == "put":
        valeur = -K*T*norm.cdf(-d2)*np.exp(-r*T)
    else:
        raise ValueError("NameError")
    return _scalaire_si_possible(np.where(regulier,valeur,0.0))


DAYS_PER_YEAR = 365.0
def theta(S, K, r, q, sigma, T, kind="call"):
    d1,d2,regulier = _d1_d2(S,K,r,q,sigma,T)
    if kind == "call":
        valeur = -S*np.exp(-q*T)*norm.pdf(d1)*sigma / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S*np.exp(-q*T)*norm.cdf(d1)
    elif kind == "put":
        valeur = -S*np.exp(-q*T)*norm.pdf(d1)*sigma / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1)
    else:
        raise ValueError("nameError")
    return _scalaire_si_possible(np.where(regulier,valeur/DAYS_PER_YEAR,0.0)) #on divise par 365 car theta est par jour calandaire



if __name__ == "__main__": #garde fou pour éviter que le code s'éexécute quand on importe black_scholes
    S0, K0, r0, q0, v0, T0 = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0
    print(f"call   = {price(S0, K0, r0, q0, v0, T0, 'call'):.6f}")
    print("Decommente la suite au fur et a mesure que tu implementes :")
    print(f"put    = {price(S0, K0, r0, q0, v0, T0, 'put'):.6f}")
    print(f"delta  = {delta(S0, K0, r0, q0, v0, T0, 'call'):.6f}")
    print(f"gamma  = {gamma(S0, K0, r0, q0, v0, T0, 'call'):.6f}")
    print(f"vega   = {vega(S0, K0, r0, q0, v0, T0, 'call'):.6f}")
    print(f"theta  = {theta(S0, K0, r0, q0, v0, T0, 'call'):.6f}  (par jour)")
    print(f"rho    = {rho(S0, K0, r0, q0, v0, T0, 'call'):.6f}")
