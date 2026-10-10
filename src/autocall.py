"""
Phoenix autocall priceur, et calcul des Greeks
"""

import numpy as np

import black_scholes as bs
import Monte_Carlo as mc
import barriere as br


# --------------------------------------------------------------------
# Dates d'observation
# --------------------------------------------------------------------
def dates_observation(T, n_obs):
    dates = np.arange(T/n_obs,T+T/n_obs,T/n_obs)
    return dates
  

# --------------------------------------------------------------------
# Pricer autocall
# --------------------------------------------------------------------
# On travail exclusivement sur des masques de boleens pour les vecteurs d'etats ici
def prix_autocall_mc(S0, r, q, sigma, T, n_obs=20, niveau_rappel=1.0,
                     barriere_coupon=0.7, barriere_protection=0.6,
                     coupon=0.02, memoire=True, protection="europeenne",
                     nominal=100.0, n_pas_par_obs=1, n_sims=100_000,
                     seed=None, S_ref = None):
    if S_ref is None: #Permet de pricer un produit qui n'est pas neuf : S_ref à 80 par ex et S0 à 100 au départ
        S_ref=S0
    # On commence par simuler un tableau de chemins sumulés pour le put barriere
    chemins = br.simuler_chemins(S0, r, q, sigma, T, n_obs*n_pas_par_obs, n_sims, seed=seed)
    #On regarde aux dates du contrat
    v_observees = chemins[:,np.arange(1,n_obs+1)*n_pas_par_obs]
    actif = np.ones(n_sims,dtype=bool)
    manques = np.zeros(n_sims,dtype=int)
    valeur_present = np.zeros(n_sims)
    dates = dates_observation(T,n_obs)
    dates_fin = np.full(n_sims,np.nan)
    proba_rappel = []
    for date_i in range(n_obs):
        df = np.exp(-r*dates[date_i]) #facteur d'actualisation pour chaque date
        paie = actif & (v_observees[:,date_i] >= barriere_coupon*S_ref)
        if memoire:
            valeur_present += df * np.where(paie, (coupon*nominal) * (1+manques), 0.0)
        else:
            valeur_present += df * np.where(paie, coupon * nominal, 0.0)
        manques = np.where(paie, 0, manques+ actif.astype(int)) #mémoire coupon que si trajectoire actif non payé

        #Cas où rappel
        rappel = actif & (v_observees[:,date_i] > S_ref * niveau_rappel)
        if date_i < n_obs-1:
            valeur_present += df * np.where(rappel, nominal, 0.0)
            dates_fin = np.where(rappel, dates[date_i], dates_fin)
            proba_rappel.append(rappel.mean())
            actif = actif & ~rappel
        else:
            proba_rappel.append(rappel.mean())

    proba_rappel = np.asarray(proba_rappel)

    # à maturité
    df_T = np.exp(-r * T)
    S_T = chemins[:, -1] 
    if protection == 'europeenne':
        casse = S_T<S_ref*barriere_protection
    elif protection == 'continue':
        minimums = chemins.min(axis=1)
        casse = (S_T<S_ref) & (minimums<S_ref*barriere_protection)
    else:
        raise ValueError('Entrez une protection "europeenne" ou "continue" svp')
    remboursement = np.where(casse, nominal*S_T/S_ref, nominal)
    valeur_present += df_T * np.where(actif, remboursement, 0.0)
    dates_fin = np.where(actif, T, dates_fin)

    err_std = np.std(valeur_present,ddof=1)/np.sqrt(n_sims)
    proba_survie_maturite = actif.mean()
    duree_vie_moyenne = np.nanmean(dates_fin)
    masque_perte = actif & (remboursement<nominal)
    proba_perte = masque_perte.mean()
    if masque_perte.any():
        perte_moyenne_si_perte = (nominal - remboursement[masque_perte]).mean()
    else:
        perte_moyenne_si_perte = 0.0

    resultats = {'prix':valeur_present.mean(),'err_std':err_std,'proba_rappel':proba_rappel,'proba_survie_maturite':proba_survie_maturite,
            'duree_vie_moyenne':duree_vie_moyenne,'proba_perte':proba_perte,'perte_moyenne_si_perte':perte_moyenne_si_perte}
    return resultats



# --------------------------------------------------------------------
# Delta et Vega par difference finies
# --------------------------------------------------------------------
def delta_autocall(S0, r, q, sigma, T, S_ref=None, h_reel=0.01, seed=0, **kwargs):
    if S_ref is None:
        S_ref = S0
    h = S0*h_reel
    plus = prix_autocall_mc(S0+h, r, q, sigma, T,seed=seed, **kwargs)["prix"]
    moins = prix_autocall_mc(S0-h, r, q, sigma, T, seed=seed, **kwargs)["prix"]
    return (plus-moins)/(2*h)

def vega_autocall(S0, r, q, sigma, T, h_vol=0.01, seed=0, **kwargs):
    plus = prix_autocall_mc(S0, r, q, sigma+h_vol, T,seed=seed, **kwargs)["prix"]
    moins = prix_autocall_mc(S0, r, q, sigma-h_vol, T,seed=seed, **kwargs)["prix"]
    return (plus-moins)/(2*h_vol)




if __name__ == "__main__":
    base = dict(r=0.03, q=0.01, sigma=0.20, T=5.0, n_obs=20,n_sims=100_000, seed=0)
    res = prix_autocall_mc(100.0, coupon=0.02, niveau_rappel=1.0,barriere_coupon=0.7, barriere_protection=0.6,**base)
    print(f"prix = {res['prix']:.4f} +/- {res['err_std']:.4f}")
    print(f"duree de vie moyenne = {res['duree_vie_moyenne']:.2f} ans")
    print(f"proba de perte       = {res['proba_perte']:.2%}")
