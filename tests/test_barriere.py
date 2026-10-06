"""
Batterie de validation
"""


import numpy as np
import pytest

import black_scholes as bs
import Monte_Carlo as mc
import barriere as br

SEUIL = 4.0

S0, K0, R0, Q0, V0, T0 = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0


# --------------------------------------------------------
# 1. LES TRAJECTOIRES
# --------------------------------------------------------
def test_forme_et_colonne_initiale():
    ch = br.simuler_chemins(S0, R0, Q0, V0, T0, n_pas=10, n_sims=7, seed=0)
    assert ch.shape == (7, 11), (
        "forme attendue (n_sims, n_pas + 1) : la date initiale fait partie "
        "du chemin")
    assert np.allclose(ch[:, 0], S0), "la premiere colonne doit valoir S"
    assert np.all(ch > 0), "une GBM ne peut jamais devenir negative"
    assert np.all(np.isfinite(ch))


def test_reproductibilite_chemins():
    a = br.simuler_chemins(S0, R0, Q0, V0, T0, 20, 50, seed=42)
    b = br.simuler_chemins(S0, R0, Q0, V0, T0, 20, 50, seed=42)
    c = br.simuler_chemins(S0, R0, Q0, V0, T0, 20, 50, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_chemins_deterministes_si_vol_nulle():
    """sigma = 0 : chaque date vaut exactement le forward de cette date."""
    n_pas = 12
    ch = br.simuler_chemins(S0, R0, Q0, 0.0, T0, n_pas, 5, seed=0)
    dt = T0 / n_pas
    attendu = S0 * np.exp((R0 - Q0) * dt * np.arange(n_pas + 1))
    assert np.allclose(ch, attendu)


@pytest.mark.parametrize("n_pas", [1, 12, 252])
def test_derniere_colonne_coherente_avec_simuler_terminal(n_pas):
    """
    La loi de S_T ne depend pas du nombre de pas
    utilises pour y arriver : c'est la propriete de Markov de la GBM.
    """
    n = 200_000
    S_T = br.simuler_chemins(S0, R0, Q0, V0, T0, n_pas, n, seed=5)[:, -1]
    ref = mc.simuler_terminal(S0, R0, Q0, V0, T0, n, seed=6)

    forward = S0 * np.exp((R0 - Q0) * T0)
    err = np.std(S_T, ddof=1) / np.sqrt(n)
    assert abs(np.mean(S_T) - forward) < SEUIL * err
    assert np.std(S_T, ddof=1) == pytest.approx(np.std(ref, ddof=1), rel=0.03)


def test_minimum_inferieur_au_terminal():
    """Verification de bon sens sur la structure du chemin."""
    ch = br.simuler_chemins(S0, R0, Q0, V0, T0, 50, 1000, seed=0)
    assert np.all(ch.min(axis=1) <= ch[:, -1])
    assert np.all(ch.max(axis=1) >= ch[:, -1])
    assert np.all(ch.min(axis=1) <= S0)


# --------------------------------------------------------
# 2. COHERENCE INTERNE DU PRICER : exacte, aucun biais possible
# --------------------------------------------------------
@pytest.mark.parametrize("B", [60.0, 80.0, 95.0])
@pytest.mark.parametrize("kind", ["put", "call"])
def test_parite_in_out_tirages_communs(B, kind):
    """
    Avec les mêmes tirages, on doit avoir in + out = vanille
    """
    n_pas, n_sims, graine = 50, 20_000, 3
    di, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-in",
                                kind, n_pas, n_sims, graine)
    do, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-out",
                                kind, n_pas, n_sims, graine)

    ch = br.simuler_chemins(S0, R0, Q0, V0, T0, n_pas, n_sims, seed=graine)
    vanille = np.mean(mc._payoff(ch[:, -1], K0, kind=kind)) * np.exp(-R0 * T0)

    assert di + do == pytest.approx(vanille, abs=1e-10), (
        "parite in-out violee. Si l'ecart est petit mais non nul, verifie que "
        "la graine est bien transmise a simuler_chemins dans les deux appels.")


@pytest.mark.parametrize("B", [60.0, 80.0])
def test_parite_in_out_contre_black_scholes(B):
    """La meme parite, mais contre la formule fermee : ecart statistique."""
    n_pas, n_sims, graine = 50, 100_000, 11
    di, e1 = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-in",
                                 "put", n_pas, n_sims, graine)
    do, e2 = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-out",
                                 "put", n_pas, n_sims, graine)
    ref = bs.price(S0, K0, R0, Q0, V0, T0, "put")
    err = np.hypot(e1, e2)
    assert abs((di + do) - ref) < SEUIL * err


def test_barriere_inatteignable():
    """B tres bas : le "in" ne s'active jamais, le "out" vaut la vanille."""
    n_pas, n_sims, graine = 50, 20_000, 1
    di, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 1.0, "down-and-in",
                                "put", n_pas, n_sims, graine)
    do, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 1.0, "down-and-out",
                                "put", n_pas, n_sims, graine)
    ch = br.simuler_chemins(S0, R0, Q0, V0, T0, n_pas, n_sims, seed=graine)
    vanille = np.mean(mc._payoff(ch[:, -1], K0, kind="put")) * np.exp(-R0 * T0)
    assert di == 0.0
    assert do == pytest.approx(vanille, abs=1e-10)


def test_barriere_touchee_des_le_depart():
    """
    B au-dessus du spot pour une barriere "down" : elle est franchie des la
    date 0. Le "in" vaut alors la vanille et le "out" vaut zero.
    Ce test ne passe QUE si la colonne initiale est incluse dans le chemin,
    ce qui est la raison d'etre de cette colonne.
    """
    n_pas, n_sims, graine = 50, 20_000, 2
    di, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 110.0, "down-and-in",
                                "put", n_pas, n_sims, graine)
    do, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 110.0, "down-and-out",
                                "put", n_pas, n_sims, graine)
    ch = br.simuler_chemins(S0, R0, Q0, V0, T0, n_pas, n_sims, seed=graine)
    vanille = np.mean(mc._payoff(ch[:, -1], K0, kind="put")) * np.exp(-R0 * T0)
    assert di == pytest.approx(vanille, abs=1e-10)
    assert do == 0.0


def test_monotonie_en_barriere():
    """
    Une barriere plus haute est plus facile a toucher pour une "down" :
    le knock-in vaut donc plus cher, le knock-out moins cher.
    Graine commune pour que la comparaison soit propre.
    """
    niveaux = [50., 60., 70., 80., 90.]
    di = [br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, b, "down-and-in",
                              "put", 50, 50_000, 7)[0] for b in niveaux]
    do = [br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, b, "down-and-out",
                              "put", 50, 50_000, 7)[0] for b in niveaux]
    assert np.all(np.diff(di) > 0), "le knock-in doit croitre avec B"
    assert np.all(np.diff(do) < 0), "le knock-out doit decroitre avec B"


@pytest.mark.parametrize("t", ["down-and-in", "down-and-out",
                               "up-and-in", "up-and-out"])
def test_les_quatre_types_sont_acceptes(t):
    p, e = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 90.0, t, "put",
                               20, 5_000, 0)
    assert np.isfinite(p) and e >= 0


def test_type_invalide_leve_une_erreur():
    with pytest.raises(ValueError):
        br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, 90.0, "Down-And-In",
                            "put", 20, 1_000, 0)


# --------------------------------------------------------
# 3. LA FORMULE FERMEE
# --------------------------------------------------------
def test_ferme_parite_in_out():
    di = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, 70.0, "down-and-in")
    do = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, 70.0, "down-and-out")
    assert di + do == pytest.approx(bs.price(S0, K0, R0, Q0, V0, T0, "put"),
                                    abs=1e-10)


def test_ferme_limite_barriere_basse():
    """B tendant vers 0 : le down-and-in n'a plus aucune valeur."""
    di = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, 1.0, "down-and-in")
    assert di == pytest.approx(0.0, abs=1e-6)


def test_ferme_limite_barriere_au_spot():
    """B tendant vers S : la barriere est presque toujours touchee."""
    di = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, 99.99, "down-and-in")
    assert di == pytest.approx(bs.price(S0, K0, R0, Q0, V0, T0, "put"),
                               rel=0.01)


def test_ferme_croissante_en_barriere():
    niveaux = np.linspace(20., 99., 40)
    di = [br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, b) for b in niveaux]
    assert np.all(np.diff(di) > 0)
    assert np.all(np.asarray(di) <= bs.price(S0, K0, R0, Q0, V0, T0, "put"))


def test_ferme_refuse_le_cas_non_implemente():
    """B > K : la formule ci-dessus ne s'applique pas. Mieux vaut une erreur
    explicite qu'un chiffre faux."""
    with pytest.raises(NotImplementedError):
        br.prix_barriere_ferme(S0, 80.0, R0, Q0, V0, T0, 90.0, "down-and-in")


# --------------------------------------------------------
# 4. CONVERGENCE DU MONTE CARLO VERS LA FORMULE FERMEE
# --------------------------------------------------------
def test_mc_converge_vers_la_formule_fermee():
    """
    Tolerance LACHE et volontairement asymetrique : avec 2000 pas il reste
    un biais de monitoring negatif de l'ordre de quelques centiemes. Ce test
    cherche un bug de formule, pas une egalite statistique.
    """
    B = 70.0
    ref = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, B, "down-and-in")
    p, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-in",
                               "put", n_pas=2000, n_sims=50_000, seed=0)
    assert p == pytest.approx(ref, rel=0.05)
    assert p < ref, (
        "le prix discret d'un knock-in doit rester SOUS le prix continu : "
        "une surveillance discrete rate des franchissements")


# --------------------------------------------------------
# 5. LE BIAIS DE MONITORING
# --------------------------------------------------------
def test_etude_biais_structure():
    res = br.etude_biais_monitoring(S0, K0, R0, Q0, V0, T0, 70.0,
                                    liste_pas=(12, 52), n_sims=20_000, seed=0)
    for cle in ("n_pas", "prix", "err_std", "biais", "reference"):
        assert cle in res, f"cle manquante : {cle}"
    assert len(res["prix"]) == 2


def test_biais_negatif_et_significatif_pour_un_knock_in():
    """
    Le resultat marquant de la semaine : meme a 252 pas, soit une
    surveillance quotidienne sur un an, le biais reste de plusieurs erreurs
    standards. La convention "un pas par jour de bourse" ne suffit pas.
    """
    res = br.etude_biais_monitoring(S0, K0, R0, Q0, V0, T0, 70.0,
                                    liste_pas=(12, 52, 252),
                                    n_sims=100_000, seed=0)
    assert np.all(res["biais"] < 0), "un knock-in discret sous-estime toujours"
    assert abs(res["biais"][-1]) > 3 * res["err_std"][-1], (
        "a 252 pas le biais doit encore depasser 3 erreurs standards")


def test_biais_decroit_avec_le_nombre_de_pas():
    res = br.etude_biais_monitoring(S0, K0, R0, Q0, V0, T0, 70.0,
                                    liste_pas=(12, 52, 252),
                                    n_sims=100_000, seed=0)
    amplitudes = np.abs(res["biais"])
    assert np.all(np.diff(amplitudes) < 0)


def test_biais_en_racine_de_n_pas():
    """
    Convergence en O(1/sqrt(n_pas)) : quadrupler les pas ne divise le biais
    que par 2. C'est BEAUCOUP plus lent qu'on ne l'imagine, et c'est ce qui
    justifie la correction de Broadie-Glasserman-Kou.
    """
    res = br.etude_biais_monitoring(S0, K0, R0, Q0, V0, T0, 70.0,
                                    liste_pas=(12, 52, 252),
                                    n_sims=200_000, seed=0)
    pente, _ = np.polyfit(np.log(res["n_pas"]),
                          np.log(np.abs(res["biais"])), 1)
    assert pente == pytest.approx(-0.5, abs=0.1), (
        f"pente mesuree {pente:.3f}, attendue -0.5")


def test_biais_positif_pour_un_knock_out():
    """Le signe s'inverse : un knock-out discret survit trop souvent."""
    res = br.etude_biais_monitoring(S0, K0, R0, Q0, V0, T0, 70.0,
                                    type_barriere="down-and-out",
                                    liste_pas=(12, 52), n_sims=100_000, seed=0)
    assert np.all(res["biais"] > 0)


# --------------------------------------------------------
# 6. LA CORRECTION DE BROADIE-GLASSERMAN-KOU
# --------------------------------------------------------
def test_correction_rapproche_la_barriere_du_spot():
    B = 70.0
    assert br.correction_broadie(B, V0, T0, 252, "down") > B
    assert br.correction_broadie(B, V0, T0, 252, "up") < B


def test_correction_tend_vers_la_barriere_quand_les_pas_augmentent():
    B = 70.0
    ecarts = [abs(br.correction_broadie(B, V0, T0, n, "down") - B)
              for n in (12, 52, 252, 5000)]
    assert np.all(np.diff(ecarts) < 0)
    assert ecarts[-1] < 0.2


def test_correction_amplitude_attendue():
    """Le deplacement vaut beta * sigma * sqrt(T/n_pas) en relatif."""
    B, n_pas = 70.0, 252
    attendu = B * np.exp(+0.5826 * V0 * np.sqrt(T0 / n_pas))
    assert br.correction_broadie(B, V0, T0, n_pas, "down") == pytest.approx(
        attendu, rel=1e-10)


def test_correction_reduit_le_biais():
    """
    La demonstration. On price avec la barriere corrigee et on verifie que
    l'ecart a la formule continue chute nettement.
    """
    B, n_pas, n_sims, graine = 70.0, 100, 100_000, 0
    ref = br.prix_barriere_ferme(S0, K0, R0, Q0, V0, T0, B, "down-and-in")
    brut, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B, "down-and-in",
                                  "put", n_pas, n_sims, graine)
    B_corr = br.correction_broadie(B, V0, T0, n_pas, "down")
    corrige, _ = br.prix_barriere_mc(S0, K0, R0, Q0, V0, T0, B_corr,
                                     "down-and-in", "put", n_pas, n_sims,
                                     graine)
    assert abs(corrige - ref) < abs(brut - ref) / 3, (
        f"brut={brut:.4f}, corrige={corrige:.4f}, reference={ref:.4f}")
