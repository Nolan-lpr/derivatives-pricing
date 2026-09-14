"""
I few validation tests for black_scholes.py
"""

import numpy as np
import pytest
import black_scholes as bs

DAYS_PER_YEAR = 365.0
CASES = [
    # S,    K,    r,     q,     sigma, T
    (100., 100., 0.00, 0.00, 0.20, 1.00),
    (100., 100., 0.03, 0.01, 0.20, 1.00),
    (100., 80., 0.03, 0.01, 0.20, 1.00),
    (100., 130., 0.03, 0.01, 0.20, 1.00),
    (100., 100., 0.05, 0.02, 0.05, 0.25),
    (100., 100., 0.05, 0.02, 0.80, 0.25),
    (100., 100., 0.05, 0.03, 0.30, 5.00),
    (50., 100., 0.02, 0.00, 0.40, 2.00),
    (200., 100., 0.02, 0.04, 0.15, 0.50),
    (100., 100., -0.005, 0.02, 0.25, 1.50),
]

KINDS = ["call", "put"]


# -------------------------------
# Test Parite call-put
# -------------------------------
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES) #permet de faire tous les tests CASES d'un coup
def test_parite_call_put(S, K, r, q, sigma, T):
    c = bs.price(S, K, r, q, sigma, T, "call")
    p = bs.price(S, K, r, q, sigma, T, "put")
    attendu = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert c - p == pytest.approx(attendu, abs=1e-10), (
        "La parite call-put est violee : cherche une erreur de signe ou "
        "d'actualisation dans le put, ou un q mal place dans d1.")


# --------------------------
# Test cas classique
# --------------------------
def test_cas_analytique_connu():
    from scipy.stats import norm
    attendu = 100.0 * (2 * norm.cdf(0.1) - 1)
    obtenu = bs.price(100., 100., 0., 0., 0.20, 1.)
    assert obtenu == pytest.approx(attendu, abs=1e-10)
    assert obtenu == pytest.approx(7.9656, abs=1e-3)

#cas ou call et put égaux
def test_symetrie_taux_nuls():  
    c = bs.price(100., 100., 0., 0., 0.30, 2., "call")
    p = bs.price(100., 100., 0., 0., 0.30, 2., "put")
    assert c == pytest.approx(p, abs=1e-10)


# ------------------
# Test de limites
# -------------------

#cas où T=0 et cas où sigma=0
def test_cas_degeneres_exacts():
    for kwargs in [dict(T=0.0, sigma=0.20), dict(T=1.0, sigma=0.0)]:
        for kind in KINDS:
            v = bs.price(100., 100., 0.03, 0.01, kwargs["sigma"], kwargs["T"], kind)
            assert np.isfinite(v), f"{kind} {kwargs} renvoie {v}" #renvoie false si NaN, Inf ou -Inf

#cas où la vol tend vers l'infini -> call tend vers S_exp(-qT)
def test_vol_tres_haute():
    c = bs.price(100., 100., 0.03, 0.01, 50.0, 1.0, "call")
    assert c == pytest.approx(100. * np.exp(-0.01), rel=1e-3)



# ------------------------
# Test des Monotonies classiques 
# ------------------------

#call croit en spot, put décroit
def test_croissance_en_spot():
    spots = np.linspace(50, 200, 61)
    c = np.array([bs.price(s, 100., 0.03, 0.01, 0.25, 1., "call") for s in spots])
    assert np.all(np.diff(c) > 0)
    p = np.array([bs.price(s, 100., 0.03, 0.01, 0.25, 1., "put") for s in spots])
    assert np.all(np.diff(p) < 0)

#call et put croissent en vol
def test_croissance_en_vol():
    vols = np.linspace(0.01, 1.5, 61)
    for kind in KINDS:
        v = np.array([bs.price(100., 100., 0.03, 0.01, s, 1., kind) for s in vols])
        assert np.all(np.diff(v) > 0), f"{kind} non croissant en vol"

#call décroit en strike
def test_decroissance_en_strike():
    strikes = np.linspace(50, 200, 61)
    c = np.array([bs.price(100., k, 0.03, 0.01, 0.25, 1., "call") for k in strikes])
    assert np.all(np.diff(c) < 0)

#un call est convexe en strike
def test_convexite_en_strike():
    strikes = np.linspace(60, 160, 51)
    c = np.array([bs.price(100., k, 0.03, 0.01, 0.25, 1., "call") for k in strikes])
    assert np.all(np.diff(c, 2) > -1e-12) #doit être positif, donc courbe convexe



# ---------------------------------------------------------------------------
# Comparaison Greeks avec difference finie centree (f(x+h))-f(x-h))/2h
# ---------------------------------------------------------------------------

def _diff_centree(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

#delta
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_delta_vs_df(S, K, r, q, sigma, T, kind):
    h = S * 1e-4
    num = _diff_centree(lambda s: bs.price(s, K, r, q, sigma, T, kind), S, h)
    assert bs.delta(S, K, r, q, sigma, T, kind) == pytest.approx(num, rel=1e-4, abs=1e-7)

#gamma
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_gamma_vs_df(S, K, r, q, sigma, T, kind):
    h = S * float(np.clip(0.1 * sigma * np.sqrt(T), 1e-3, 1e-2)) #le pas est volontairement plus grand car c'est une dérivée 2nd
    f = lambda s: bs.price(s, K, r, q, sigma, T, kind)
    num = (f(S + h) - 2 * f(S) + f(S - h)) / h ** 2
    assert bs.gamma(S, K, r, q, sigma, T, kind) == pytest.approx(num, rel=1e-3, abs=1e-7)

#vega
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_vega_vs_df(S, K, r, q, sigma, T, kind):
    h = 1e-4
    num = _diff_centree(lambda v: bs.price(S, K, r, q, v, T, kind), sigma, h)
    assert bs.vega(S, K, r, q, sigma, T, kind) == pytest.approx(num, rel=1e-4, abs=1e-7)

#theta
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_theta_vs_df(S, K, r, q, sigma, T, kind):
    h = min(1e-4, T / 10)
    num = -_diff_centree(lambda t: bs.price(S, K, r, q, sigma, t, kind), T, h)
    attendu = num / DAYS_PER_YEAR
    assert bs.theta(S, K, r, q, sigma, T, kind) == pytest.approx(attendu, rel=1e-3, abs=1e-9)

#rho
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_rho_vs_df(S, K, r, q, sigma, T, kind):
    h = 1e-6
    num = _diff_centree(lambda x: bs.price(S, K, x, q, sigma, T, kind), r, h)
    assert bs.rho(S, K, r, q, sigma, T, kind) == pytest.approx(num, rel=1e-4, abs=1e-6)



#-------------------------------
#   COHERENCE GLOBALE GREEKS
#-------------------------------


#vega call/put identiques
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
def test_vega_identique_call_put(S, K, r, q, sigma, T):
    assert bs.vega(S, K, r, q, sigma, T, "call") == pytest.approx(
        bs.vega(S, K, r, q, sigma, T, "put"), rel=1e-12)

#gamma call/put identique
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
def test_gamma_identique_call_put(S, K, r, q, sigma, T):
    assert bs.gamma(S, K, r, q, sigma, T, "call") == pytest.approx(
        bs.gamma(S, K, r, q, sigma, T, "put"), rel=1e-12)

#delta call/put écarté de e^(-qT)
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
def test_ecart_des_deltas(S, K, r, q, sigma, T):
    dc = bs.delta(S, K, r, q, sigma, T, "call")
    dp = bs.delta(S, K, r, q, sigma, T, "put")
    assert dc - dp == pytest.approx(np.exp(-q * T), abs=1e-10)

#signes des greeks attendus
@pytest.mark.parametrize("S,K,r,q,sigma,T", CASES)
def test_signes_des_greeks(S, K, r, q, sigma, T):
    assert 0.0 <= bs.delta(S, K, r, q, sigma, T, "call") <= np.exp(-q * T) + 1e-12
    assert -np.exp(-q * T) - 1e-12 <= bs.delta(S, K, r, q, sigma, T, "put") <= 0.0
    assert bs.gamma(S, K, r, q, sigma, T, "call") >= 0.0
    assert bs.vega(S, K, r, q, sigma, T, "call") >= 0.0
    assert bs.rho(S, K, r, q, sigma, T, "call") >= 0.0
    assert bs.rho(S, K, r, q, sigma, T, "put") <= 0.0
