"""
Batterie de validation pour autocall.py

Lancer :  pytest -v tests/test_autocall.py   (depuis la racine du projet)

--------------------------------------------------------------------------
LE PROBLEME DE CE MODULE : IL N'Y A PLUS DE REFERENCE
--------------------------------------------------------------------------
Pour la vanille tu avais Black-Scholes, pour la barriere tu avais
Reiner-Rubinstein. Pour un phoenix autocall, aucune formule fermee n'existe.
Tu ne peux donc plus valider le prix directement.

La strategie qui remplace la reference : les CAS DEGENERES. On choisit des
parametres qui reduisent l'autocall a un produit dont on connait le prix
exactement, et on verifie que le pricer retombe dessus.

  - coupon nul, rappel impossible, protection jamais touchee
        -> le produit est une obligation zero coupon : nominal * exp(-rT)
           EXACT, aucune erreur statistique, puisque tous les chemins
           paient la meme chose

  - coupon nul, rappel impossible, protection europeenne a 100 %
        -> le produit est une obligation zero coupon MOINS un put vanille
           de strike S0 : verifiable contre bs.price

  - rappel au premier passage, coupon toujours paye
        -> le produit est un flux unique a la premiere date
           EXACT la aussi

Ces trois cas encadrent toute la mecanique : les coupons, le rappel, la
protection. Si les trois passent, l'essentiel de la boucle est juste.

Le reste des tests porte sur la COHERENCE INTERNE : les probabilites
somment a un, la duree de vie est bornee, les monotonies ont le bon sens.

--------------------------------------------------------------------------
CONTRAT SUPPOSE
--------------------------------------------------------------------------
    dates_observation(T, n_obs) -> tableau des n_obs dates, la derniere = T

    prix_autocall_mc(S0, r, q, sigma, T, n_obs, niveau_rappel,
                     barriere_coupon, barriere_protection, coupon,
                     memoire, protection, nominal, n_pas_par_obs,
                     n_sims, seed)
        -> dict : "prix", "err_std", "proba_rappel" (un par date),
                  "proba_survie_maturite", "duree_vie_moyenne",
                  "proba_perte", "perte_moyenne_si_perte"

    Les niveaux (rappel, coupon, protection) sont exprimes en FRACTION de
    S0, pas en valeur absolue : 0.6 veut dire 60 % du niveau initial.
    Le coupon est une fraction du nominal, PAR DATE d'observation.
"""
import numpy as np
import pytest

import black_scholes as bs
import autocall as ac

SEUIL = 4.0

S0, R0, Q0, V0, T0 = 100.0, 0.03, 0.01, 0.20, 5.0
N_OBS = 20
NOMINAL = 100.0

BASE = dict(r=R0, q=Q0, sigma=V0, T=T0, n_obs=N_OBS,
            n_sims=100_000, seed=0)


# ===========================================================================
# 1. LES DATES
# ===========================================================================
def test_dates_observation():
    d = ac.dates_observation(T0, N_OBS)
    assert len(d) == N_OBS
    assert d[-1] == pytest.approx(T0)
    assert np.all(np.diff(d) > 0)
    assert d[0] == pytest.approx(T0 / N_OBS)


# ===========================================================================
# 2. LES TROIS CAS DEGENERES : le coeur de la validation
# ===========================================================================
def test_obligation_zero_coupon():
    """
    Coupon nul, rappel impossible, protection jamais touchee : tous les
    chemins paient exactement le nominal a maturite. Le prix est donc le
    nominal actualise, SANS aucune erreur statistique.

    Si ce test echoue, le probleme est dans l'actualisation ou dans le
    remboursement final, pas dans la boucle d'observation.
    """
    res = ac.prix_autocall_mc(S0, coupon=0.0, niveau_rappel=10.0,
                              barriere_protection=0.0, **BASE)
    assert res["prix"] == pytest.approx(NOMINAL * np.exp(-R0 * T0), abs=1e-8)
    assert res["err_std"] == pytest.approx(0.0, abs=1e-12), (
        "tous les chemins paient la meme chose : la dispersion doit etre nulle")
    assert res["proba_survie_maturite"] == 1.0
    assert res["proba_perte"] == 0.0


def test_zero_coupon_moins_un_put():
    """
    Meme chose, mais avec la protection a 100 % de S0. Le porteur recoit
    alors le nominal si S_T >= S0, et nominal * S_T / S0 sinon, ce qui
    s'ecrit nominal moins le payoff d'un put de strike S0 mis a l'echelle.

    C'est LE test qui valide la jambe optionnelle. Il montre aussi,
    concretement, ce que l'investisseur est : long une obligation, court
    un put.
    """
    res = ac.prix_autocall_mc(S0, coupon=0.0, niveau_rappel=10.0,
                              barriere_protection=1.0, **BASE)
    attendu = (NOMINAL * np.exp(-R0 * T0)
               - (NOMINAL / S0) * bs.price(S0, S0, R0, Q0, V0, T0, "put"))
    assert abs(res["prix"] - attendu) < SEUIL * res["err_std"], (
        f"MC={res['prix']:.4f} +/- {res['err_std']:.4f}, attendu {attendu:.4f}")


def test_rappel_immediat():
    """
    Niveau de rappel a zero et barriere de coupon a zero : toute trajectoire
    declenche le rappel des la premiere date et touche son coupon. Le produit
    se reduit a un flux unique certain.

    Ce test valide la date de paiement : une erreur d'un cran dans les
    indices d'observation se voit immediatement sur l'actualisation.
    """
    coupon = 0.02
    res = ac.prix_autocall_mc(S0, coupon=coupon, niveau_rappel=0.0,
                              barriere_coupon=0.0, barriere_protection=0.0,
                              **BASE)
    t1 = T0 / N_OBS
    attendu = NOMINAL * (1 + coupon) * np.exp(-R0 * t1)
    assert res["prix"] == pytest.approx(attendu, abs=1e-8)
    assert res["proba_rappel"][0] == 1.0
    assert res["duree_vie_moyenne"] == pytest.approx(t1)


# ===========================================================================
# 3. COHERENCE DES METRIQUES DE DISTRIBUTION
# ===========================================================================
from scipy.stats import norm
@pytest.fixture(scope="module")
def produit():
    """Un phoenix realiste, calcule une seule fois pour tous les tests."""
    return ac.prix_autocall_mc(S0, coupon=0.02, niveau_rappel=1.0,
                               barriere_coupon=0.7, barriere_protection=0.6,
                               **BASE)


def test_probabilites_bien_formees(produit):
    p = produit["proba_rappel"]
    assert np.all(p >= 0) and np.all(p <= 1)
    assert 0 <= produit["proba_survie_maturite"] <= 1
    assert 0 <= produit["proba_perte"] <= 1


def test_les_probabilites_somment_a_un(produit):
    """
    Toute trajectoire finit quelque part : rappelee a l'une des dates
    intermediaires, ou portee jusqu'a maturite. Si cette somme ne vaut pas
    un, une trajectoire est comptee deux fois ou perdue en route, ce qui est
    le bug le plus frequent dans une boucle de payoff a etats.
    """
    total = produit["proba_rappel"][:-1].sum() + produit["proba_survie_maturite"]
    assert total == pytest.approx(1.0, abs=1e-12)


def test_duree_de_vie_bornee(produit):
    assert 0 < produit["duree_vie_moyenne"] <= T0


def test_rappel_plus_probable_a_la_premiere_date(produit):
    """
    La probabilite de rappel decroit fortement apres la premiere date :
    celle-ci absorbe toutes les trajectoires au-dessus du niveau, les
    suivantes ne voient que les survivantes. C'est ce qui donne a l'autocall
    sa duree de vie courte malgre une maturite longue.
    """
    assert produit["proba_rappel"][0] > produit["proba_rappel"][1]
    assert produit["duree_vie_moyenne"] < T0 / 2


def test_perte_seulement_a_maturite(produit):
    """Une trajectoire rappelee recupere son nominal : elle ne peut pas
    perdre. La probabilite de perte est donc bornee par la survie."""
    assert produit["proba_perte"] <= produit["proba_survie_maturite"]


# ===========================================================================
# 4. MONOTONIES : le sens economique
# ===========================================================================
def test_prix_croissant_en_coupon():
    prix = [ac.prix_autocall_mc(S0, coupon=c, niveau_rappel=1.0,
                                barriere_coupon=0.7, barriere_protection=0.6,
                                **BASE)["prix"]
            for c in (0.0, 0.01, 0.02, 0.03)]
    assert np.all(np.diff(prix) > 0)


def test_prix_decroissant_en_barriere_de_protection():
    """
    Une barriere de protection plus haute est plus facile a casser, donc le
    porteur est plus expose : la note vaut moins cher.
    """
    prix = [ac.prix_autocall_mc(S0, coupon=0.02, niveau_rappel=1.0,
                                barriere_coupon=0.7, barriere_protection=b,
                                **BASE)["prix"]
            for b in (0.4, 0.6, 0.8, 1.0)]
    assert np.all(np.diff(prix) < 0)


def test_prix_decroissant_en_barriere_de_coupon():
    """Une barriere de coupon plus haute rend le coupon plus difficile a
    obtenir."""
    prix = [ac.prix_autocall_mc(S0, coupon=0.02, niveau_rappel=1.0,
                                barriere_coupon=b, barriere_protection=0.6,
                                **BASE)["prix"]
            for b in (0.5, 0.7, 0.9)]
    assert np.all(np.diff(prix) < 0)


def test_memoire_augmente_le_prix():
    """
    Le coupon memoire ne peut que rajouter des flux : a parametres egaux et
    MEME GRAINE, la version avec memoire vaut au moins autant que sans.
    La comparaison n'a de sens qu'a tirages communs, sinon l'ecart serait
    noye dans le bruit.
    """
    args = dict(coupon=0.02, niveau_rappel=1.0, barriere_coupon=0.7,
                barriere_protection=0.6, **BASE)
    avec = ac.prix_autocall_mc(S0, memoire=True, **args)["prix"]
    sans = ac.prix_autocall_mc(S0, memoire=False, **args)["prix"]
    assert avec > sans


def test_protection_continue_moins_chere_que_europeenne():
    """
    Une barriere surveillee en continu est plus facile a casser qu'une
    barriere regardee seulement a maturite. Le porteur est donc plus expose
    et la note vaut moins. C'est la distinction a savoir faire en entretien
    entre une protection europeenne et une protection americaine.
    """
    args = dict(coupon=0.02, niveau_rappel=1.0, barriere_coupon=0.7,
                barriere_protection=0.6, n_pas_par_obs=10, **BASE)
    euro = ac.prix_autocall_mc(S0, protection="europeenne", **args)["prix"]
    cont = ac.prix_autocall_mc(S0, protection="continue", **args)["prix"]
    assert cont < euro


# ===========================================================================
# 5. ROBUSTESSE
# ===========================================================================
def test_reproductibilite():
    args = dict(coupon=0.02, niveau_rappel=1.0, barriere_coupon=0.7,
                barriere_protection=0.6, **BASE)
    a = ac.prix_autocall_mc(S0, **args)
    b = ac.prix_autocall_mc(S0, **args)
    assert a["prix"] == b["prix"]
    assert a["err_std"] == b["err_std"]


def test_protection_invalide_leve_une_erreur():
    with pytest.raises(ValueError):
        ac.prix_autocall_mc(S0, protection="americaine", coupon=0.02,
                            niveau_rappel=1.0, barriere_coupon=0.7,
                            barriere_protection=0.6, **BASE)


def test_prix_dans_des_bornes_raisonnables(produit):
    """Un garde-fou grossier : une note phoenix ne vaut ni zero ni le
    double du nominal."""
    assert 0.5 * NOMINAL < produit["prix"] < 1.5 * NOMINAL


# ===========================================================================
# 6. A TOI D'ECRIRE
# ===========================================================================
def test_sens_de_la_monotonie_en_niveau_de_rappel():
    """
    TODO A : deux effets s'opposent. Un rappel plus difficile allonge la vie
    du produit, donc rapporte plus de coupons, mais expose plus longtemps a
    la barriere de protection. Lequel gagne depend entierement du coupon.

    Avec 2 % par trimestre, soit 8 % par an contre un taux sans risque de
    3 %, le flux de coupons est tres riche : rester vivant est une bonne
    nouvelle, le prix monte avec le niveau de rappel.

    Coupon nul, l'arbitrage s'inverse : il n'y a plus rien a gagner a rester
    vivant, et un rappel precoce rend le nominal plus tot, donc moins
    actualise. Le prix baisse avec le niveau de rappel.
    """
    args = dict(barriere_coupon=0.7, barriere_protection=0.6, **BASE)

    riche = [ac.prix_autocall_mc(S0, coupon=0.02, niveau_rappel=n, **args)["prix"]
             for n in (0.9, 1.0, 1.1, 1.3)]
    assert np.all(np.diff(riche) > 0), f"coupon riche : {riche}"

    nul = [ac.prix_autocall_mc(S0, coupon=0.0, niveau_rappel=n, **args)["prix"]
           for n in (0.9, 1.0, 1.1, 1.3)]
    assert np.all(np.diff(nul) < 0), f"coupon nul : {nul}"


def test_sens_de_la_monotonie_en_volatilite():
    """
    TODO B : le porteur a vendu un put down-and-in, il est donc COURT de
    volatilite. Contrairement a une vanille, le prix BAISSE quand la vol
    monte. C'est la reponse attendue en entretien.

    Avec une protection tres basse, le put vendu est presque sans valeur :
    la sensibilite a la vol ne s'inverse pas vraiment, elle s'effondre. On
    verifie donc que l'amplitude devient beaucoup plus faible, pas que le
    signe change.
    """
    base_sans_vol = {k: v for k, v in BASE.items() if k != "sigma"}
    args = dict(coupon=0.02, niveau_rappel=1.0, barriere_coupon=0.7,
                **base_sans_vol)

    vols = (0.15, 0.20, 0.25, 0.30)
    expose = [ac.prix_autocall_mc(S0, sigma=v, barriere_protection=0.6, **args)["prix"]
              for v in vols]
    assert np.all(np.diff(expose) < 0), f"protection 60 % : {expose}"

    protege = [ac.prix_autocall_mc(S0, sigma=v, barriere_protection=0.2, **args)["prix"]
               for v in vols]
    amplitude_expose = max(expose) - min(expose)
    amplitude_protege = max(protege) - min(protege)
    assert amplitude_protege < amplitude_expose / 3, (
        f"amplitudes : exposee {amplitude_expose:.3f}, "
        f"protegee {amplitude_protege:.3f}")


def test_delta_par_tirages_communs():
    """
    TODO C : aucune formule fermee pour le delta d'un autocall. Differences
    finies centrees, avec le spot courant decale et le niveau de reference
    contractuel FIXE (sinon le produit est invariant d'echelle et le delta
    vaut zero identiquement).

    Le payoff est discontinu : une trajectoire qui passe d'un cote ou de
    l'autre du niveau de rappel fait sauter le prix d'un bloc entier. Sans
    tirages communs, ce bruit noie completement la pente. C'est ici que la
    technique passe de confortable a indispensable.
    """
    args = dict(coupon=0.02, niveau_rappel=1.0, barriere_coupon=0.7,
                barriere_protection=0.6, S_ref=S0, n_obs=N_OBS,
                r=R0, q=Q0, sigma=V0, T=T0, n_sims=20_000)
    spot, h = 80.0, 0.8

    communs, independants = [], []
    for s in range(10):
        haut = ac.prix_autocall_mc(spot + h, seed=s, **args)["prix"]
        bas_c = ac.prix_autocall_mc(spot - h, seed=s, **args)["prix"]
        bas_i = ac.prix_autocall_mc(spot - h, seed=s + 1000, **args)["prix"]
        communs.append((haut - bas_c) / (2 * h))
        independants.append((haut - bas_i) / (2 * h))

    assert np.mean(communs) > 0, (
        "loin sous le niveau initial, le put vendu domine : delta positif")
    assert np.std(communs) < np.std(independants) / 5, (
        f"ecarts-types : communs {np.std(communs):.4f}, "
        f"independants {np.std(independants):.4f}")


def test_somme_des_flux_actualises():
    """
    TODO D : test de conservation exact. Coupon nul, rappel impossible,
    protection europeenne a un niveau B. Le porteur recoit le nominal sauf
    si S_T < B, auquel cas il recoit nominal * S_T / S0.

    La jambe optionnelle vaut donc (S0 - S_T) quand S_T < B, ce qui se
    decompose proprement :

        (S0 - S_T) 1{S_T<B} = (B - S_T)+ + (S0 - B) 1{S_T<B}

    soit un put de strike B, plus (S0 - B) digitales put de barriere B. Les
    deux ont une forme fermee, donc le prix attendu est EXACT.

    Cette decomposition est exactement celle qu'un structureur fait au
    tableau pour expliquer ou part la perte du client.
    """
    B_rel = 0.6
    B = B_rel * S0
    res = ac.prix_autocall_mc(S0, coupon=0.0, niveau_rappel=10.0,
                              barriere_protection=B_rel,
                              protection="europeenne", **BASE)

    d1 = (np.log(S0 / B) + (R0 - Q0 + 0.5 * V0**2) * T0) / (V0 * np.sqrt(T0))
    d2 = d1 - V0 * np.sqrt(T0)
    digitale_put = np.exp(-R0 * T0) * norm.cdf(-d2)
    jambe = bs.price(S0, B, R0, Q0, V0, T0, "put") + (S0 - B) * digitale_put

    attendu = NOMINAL * np.exp(-R0 * T0) - (NOMINAL / S0) * jambe
    assert abs(res["prix"] - attendu) < SEUIL * res["err_std"], (
        f"MC={res['prix']:.4f} +/- {res['err_std']:.4f}, attendu {attendu:.4f}")