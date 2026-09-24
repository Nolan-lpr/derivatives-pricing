"""
Batterie de validation pour Monte_Carlo.py (graine fixée, tolérance statistique et non absolue)
avec pytest -v tests/test_monte_carlo.py
"""

import numpy as np
import pytest
import black_scholes as bs
import Monte_Carlo as mc

SEUIL = 4.0          # nombre d'erreurs standards tolerees
N_GRAND = 400_000    # echantillon des tests de biais

# S, K, r, q, sigma, T
CAS = [
    (100., 100., 0.03, 0.01, 0.20, 1.00),
    (100., 80., 0.03, 0.01, 0.20, 1.00),
    (100., 130., 0.03, 0.01, 0.30, 2.00),
    (100., 100., 0.05, 0.03, 0.15, 0.25),
    (50., 100., 0.02, 0.00, 0.40, 2.00),
]
KINDS = ["call", "put"]


# --------------------------------------------------------
# La simulation des chemins (simuler_terminal)
# --------------------------------------------------------

#Sous la mesure risque-neutre, l'esperance de S_T est le forward.
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
def test_identite_martingale(S, K, r, q, sigma, T):
    S_T = mc.simuler_terminal(S, r, q, sigma, T, N_GRAND, seed=7)
    forward = S * np.exp((r - q) * T)
    err = np.std(S_T, ddof=1) / np.sqrt(N_GRAND)
    ecart = abs(np.mean(S_T) - forward)
    assert ecart < SEUIL * err, (
        f"moyenne={np.mean(S_T):.4f}, forward={forward:.4f}, "
        f"ecart={ecart:.4f}, seuil={SEUIL*err:.4f}. "
        "Verifie le terme -sigma^2/2 et le placement du *T.")

#On verifie aussi le second moment : Var(S_T) = S^2 e^{2(r-q)T} (e^{sigma^2 T} - 1).
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
def test_loi_lognormale_variance(S, K, r, q, sigma, T):
    S_T = mc.simuler_terminal(S, r, q, sigma, T, N_GRAND, seed=11)
    var_theo = S**2 * np.exp(2*(r-q)*T) * (np.exp(sigma**2 * T) - 1)
    assert np.var(S_T, ddof=1) == pytest.approx(var_theo, rel=0.05)

#log-rendement doit etre gaussien de moyenne (r-q-sigma^2/2)T et d'ecart type sigma*sqrt(T)
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
def test_log_rendements_gaussiens(S, K, r, q, sigma, T):
    S_T = mc.simuler_terminal(S, r, q, sigma, T, N_GRAND, seed=13)
    logret = np.log(S_T / S)
    mu_theo = (r - q - 0.5 * sigma**2) * T
    sd_theo = sigma * np.sqrt(T)
    err = sd_theo / np.sqrt(N_GRAND)
    assert abs(np.mean(logret) - mu_theo) < SEUIL * err
    assert np.std(logret, ddof=1) == pytest.approx(sd_theo, rel=0.02)


def test_forme_et_positivite():
    S_T = mc.simuler_terminal(100., 0.03, 0.01, 0.20, 1.0, 5_000, seed=0)
    assert np.asarray(S_T).shape == (5_000,)
    assert np.all(S_T > 0), "une GBM ne peut jamais devenir negative"
    assert np.all(np.isfinite(S_T))


def test_reproductibilite():
    a = mc.simuler_terminal(100., 0.03, 0.01, 0.20, 1.0, 1_000, seed=42)
    b = mc.simuler_terminal(100., 0.03, 0.01, 0.20, 1.0, 1_000, seed=42)
    c = mc.simuler_terminal(100., 0.03, 0.01, 0.20, 1.0, 1_000, seed=43)
    assert np.array_equal(a, b), "meme graine, resultat different : rng mal place"
    assert not np.array_equal(a, c), "graines differentes, resultat identique"


def test_cas_deterministes():
    # sigma = 0 : plus d'alea, S_T vaut exactement le forward
    S_T = mc.simuler_terminal(100., 0.03, 0.01, 0.0, 1.0, 1_000, seed=0)
    assert np.allclose(S_T, 100. * np.exp((0.03 - 0.01) * 1.0))
    # T = 0 : rien ne s'est passe
    S_T = mc.simuler_terminal(100., 0.03, 0.01, 0.20, 0.0, 1_000, seed=0)
    assert np.allclose(S_T, 100.)


# --------------------------------------------------------
# le pricer : prix_mc
# --------------------------------------------------------
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
@pytest.mark.parametrize("kind", KINDS)
def test_prix_converge_vers_black_scholes(S, K, r, q, sigma, T, kind):
    prix, err = mc.prix_mc(S, K, r, q, sigma, T, kind, N_GRAND, seed=3)
    ref = bs.price(S, K, r, q, sigma, T, kind)
    assert abs(prix - ref) < SEUIL * err, (
        f"MC={prix:.4f} +/- {err:.4f}, BS={ref:.4f}, "
        f"ecart={abs(prix-ref):.4f}")


def test_decroissance_de_l_erreur_en_racine_de_n():
    """Multiplier N par 100 doit diviser l'erreur standard par ~10."""
    _, e_petit = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                            10_000, seed=1)
    _, e_grand = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                            1_000_000, seed=1)
    assert e_petit / e_grand == pytest.approx(10.0, rel=0.15)


@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
def test_parite_call_put_tirages_communs(S, K, r, q, sigma, T):
    n = 50_000
    c, _ = mc.prix_mc(S, K, r, q, sigma, T, "call", n, seed=9)
    p, _ = mc.prix_mc(S, K, r, q, sigma, T, "put", n, seed=9)
    S_T = mc.simuler_terminal(S, r, q, sigma, T, n, seed=9)

    # identite algebrique : exacte
    assert c - p == pytest.approx(np.exp(-r * T) * (np.mean(S_T) - K), abs=1e-8)

    # parite theorique : statistique seulement
    attendu = S * np.exp(-q * T) - K * np.exp(-r * T)
    err = np.exp(-r * T) * np.std(S_T, ddof=1) / np.sqrt(n)
    assert abs((c - p) - attendu) < SEUIL * err


def test_prix_mc_reproductible():
    a = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call", 10_000, seed=4)
    b = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call", 10_000, seed=4)
    assert a == b


# --------------------------------------------------------
# La convergence : etude_convergence
# --------------------------------------------------------
def test_etude_convergence_structure():
    res = mc.etude_convergence(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                               tailles=(10**3, 10**4, 10**5), seed=0)
    for cle in ("tailles", "prix", "err_std", "erreur_reelle", "reference"):
        assert cle in res, f"cle manquante : {cle}"
    assert len(res["prix"]) == 3
    assert res["reference"] == pytest.approx(
        bs.price(100., 100., 0.03, 0.01, 0.20, 1.0, "call"))

#En echelle log-log, l'erreur standard en fonction de N est une droite de pente -1/2.
def test_pente_log_log_vaut_moins_un_demi():
    res = mc.etude_convergence(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                               tailles=(10**3, 10**4, 10**5, 10**6), seed=0)
    pente, _ = np.polyfit(np.log(np.asarray(res["tailles"], dtype=float)),
                          np.log(np.asarray(res["err_std"], dtype=float)), 1)
    assert pente == pytest.approx(-0.5, abs=0.05), (
        f"pente mesuree={pente:.3f}, attendue -0.5")


def test_erreur_reelle_diminue_globalement():
    res = mc.etude_convergence(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                               tailles=(10**3, 10**6), seed=0)
    assert res["erreur_reelle"][-1] < res["erreur_reelle"][0]


# --------------------------------------------------------
# Réduction de variance : antithetique
# --------------------------------------------------------
#Reduire la variance ne doit RIEN changer au prix estime.
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
@pytest.mark.parametrize("kind", KINDS)
def test_antithetique_sans_biais(S, K, r, q, sigma, T, kind):
    prix, err = mc.prix_mc_antithetique(S, K, r, q, sigma, T, kind,
                                        N_GRAND, seed=21)
    ref = bs.price(S, K, r, q, sigma, T, kind)
    assert abs(prix - ref) < SEUIL * err, (
        f"anti={prix:.4f} +/- {err:.4f}, BS={ref:.4f}")


def test_antithetique_reduit_la_variance():
    """
    A BUDGET DE TIRAGES EGAL. Comparer 100 000 paires contre 100 000 tirages
    simples serait tricher : c'est deux fois plus de travail.
    """
    n = 200_000
    _, e_simple = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                             n, seed=31)
    _, e_anti = mc.prix_mc_antithetique(100., 100., 0.03, 0.01, 0.20, 1.0,
                                        "call", n, seed=31)
    assert e_anti < e_simple, (
        f"err simple={e_simple:.5f}, err antithetique={e_anti:.5f} : "
        "aucun gain. Verifie que tu moyennes par PAIRE avant de calculer "
        "l'ecart type.")



# --------------------------------------------------------
# réduction de variance : variable de controle
# --------------------------------------------------------
@pytest.mark.parametrize("S,K,r,q,sigma,T", CAS)
@pytest.mark.parametrize("kind", KINDS)
def test_variable_controle_sans_biais(S, K, r, q, sigma, T, kind):
    prix, err = mc.prix_mc_variable_controle(S, K, r, q, sigma, T, kind,
                                             N_GRAND, seed=23)
    ref = bs.price(S, K, r, q, sigma, T, kind)
    assert abs(prix - ref) < SEUIL * err, (
        f"vc={prix:.4f} +/- {err:.4f}, BS={ref:.4f}")


def test_variable_controle_reduit_la_variance():
    n = 200_000
    _, e_simple = mc.prix_mc(100., 100., 0.03, 0.01, 0.20, 1.0, "call",
                             n, seed=37)
    _, e_vc = mc.prix_mc_variable_controle(100., 100., 0.03, 0.01, 0.20, 1.0,
                                           "call", n, seed=37)
    assert e_vc < e_simple
    facteur = (e_simple / e_vc) ** 2
    assert facteur > 1.5, (
        f"facteur de reduction de variance={facteur:.2f} : trop faible. "
        "Verifie l'estimation de beta = Cov(payoff, S_T) / Var(S_T).")


#La méthode de variable de contrôle est très efficace dans la monnaie
def test_variable_controle_tres_efficace_dans_la_monnaie():
    n = 100_000
    _, e_simple = mc.prix_mc(100., 50., 0.03, 0.01, 0.20, 1.0, "call",
                             n, seed=41)
    _, e_vc = mc.prix_mc_variable_controle(100., 50., 0.03, 0.01, 0.20, 1.0,
                                           "call", n, seed=41)
    assert (e_simple / e_vc) ** 2 > 20


# --------------------------------------------------------
# Cohérence d'ensemble
# --------------------------------------------------------
@pytest.mark.parametrize("kind", KINDS)
def test_les_trois_methodes_donnent_le_meme_prix(kind):
    """
    Trois estimateurs differents du meme nombre. Leurs intervalles de
    confiance doivent se recouvrir, sinon l'un d'eux est biaise.
    """
    args = (100., 105., 0.03, 0.01, 0.20, 1.0, kind, 200_000)
    p1, e1 = mc.prix_mc(*args, seed=51)
    p2, e2 = mc.prix_mc_antithetique(*args, seed=51)
    p3, e3 = mc.prix_mc_variable_controle(*args, seed=51)
    ref = bs.price(100., 105., 0.03, 0.01, 0.20, 1.0, kind)
    for nom, p, e in [("simple", p1, e1), ("anti", p2, e2), ("controle", p3, e3)]:
        assert abs(p - ref) < SEUIL * e, f"{nom} : {p:.4f} +/- {e:.4f} vs {ref:.4f}"
