'''Pricer d'options barrières'''



import numpy as np
from scipy.stats import norm 

import black_scholes as bs
import Monte_Carlo as mc




# -------------------------------------------------------------
# Simulation des trajectoires
# -------------------------------------------------------------
# On commence par créer un tableau [n_simulation x n+1_ observations] qui décrit les chemins
def simuler_chemins(S, r, q, sigma, T, n_pas, n_sims, seed=None):
    rng = np.random.default_rng(seed)
    dt = T/n_pas
    Z = rng.standard_normal((n_sims,n_pas))
    drift = (r-q-sigma**2/2)*dt
    vol = sigma*np.sqrt(dt)*Z
    matrice = drift+vol
    cumul = np.cumsum(matrice,axis=1)
    chemins = S*np.exp(cumul)
    tableau_S_initial = np.full((n_sims,1),S)
    chemins_simulees = np.hstack([tableau_S_initial,chemins])
    return chemins_simulees
    


# -------------------------------------------------------------
# Pricer de l'option barrière et erreur standard
# -------------------------------------------------------------
# On price les barrières en fonction des 4 scénarios
def prix_barriere_mc(S, K, r, q, sigma, T, B, type_barriere="down-and-in",
                     kind="put", n_pas=252, n_sims=100_000, seed=None):
    chemins = simuler_chemins(S, r, q, sigma, T, n_pas, n_sims, seed=seed)
    S_T = chemins[:,-1]
    if type_barriere in("down-and-in", "down-and-out"):
        minimums = chemins.min(axis=1)
        touchees = minimums<=B
        if type_barriere=="down-and-in":
            payoff_final = np.where(touchees,mc._payoff(S_T,K,kind=kind),0.0)
        if type_barriere=="down-and-out":
            payoff_final = np.where(~touchees,mc._payoff(S_T,K,kind=kind),0.0)            
    elif type_barriere in ("up-and-in", "up-and-out"):
        maximums = chemins.max(axis=1)
        touchees = maximums>=B
        if type_barriere=="up-and-in":
            payoff_final = np.where(touchees,mc._payoff(S_T,K,kind=kind),0.0)
        if type_barriere=="up-and-out":
            payoff_final = np.where(~touchees,mc._payoff(S_T,K,kind=kind),0.0)
    else: 
        raise ValueError("Error : type_barriere should be 'down-and-in' or 'down-and-out' or 'up-and-in' 'up-and-out'")  
    payoff_act = payoff_final*np.exp(-r*T)
    moyenne = np.mean(payoff_act)
    erreur_standard = np.std(payoff_act,ddof=1)/np.sqrt(n_sims)
    return (moyenne,erreur_standard)
    
  


# -------------------------------------------------------------
# Forme fermée de Reiner-Rubenstein (pour un put down-and-in ici)
# -------------------------------------------------------------
# On n'a fait seulement le cas down-and-in avec B<=K (et out par parité), car les autres cas ne nous intéressent pas pour l'autocall
def prix_barriere_ferme(S, K, r, q, sigma, T, B, type_barriere="down-and-in", kind="put"):
    lambd = (r-q+sigma**2/2)/sigma**2
    y = np.log(B**2/(S*K)) /(sigma*np.sqrt(T))  +lambd*sigma*np.sqrt(T)
    x1 = np.log(S/B)/(sigma*np.sqrt(T)) + lambd*sigma*np.sqrt(T)
    y1 = np.log(B/S)/(sigma*np.sqrt(T)) + lambd*sigma*np.sqrt(T)
    price_di = (
            -S*np.exp(-q*T)*norm.cdf(-x1) + K*np.exp(-r*T)*norm.cdf(-x1+sigma*np.sqrt(T))
              +S*np.exp(-q*T)*((B/S)**(2*lambd))*(norm.cdf(y)-norm.cdf(y1))
              - K*np.exp(-r*T)*((B/S)**(2*lambd-2))*(norm.cdf(y-sigma*np.sqrt(T))-norm.cdf(y1-sigma*np.sqrt(T)))
        )
    if type_barriere=="down-and-in" and B<=K:
        price = price_di
    elif type_barriere=="down-and-out" and B<=K:
        price = bs.price(S,K,r,q,sigma,T,kind=kind) - price_di

    elif type_barriere=="down-and-in" and B>K:
        raise NotImplementedError("Error : Not Implemented")
    elif type_barriere=="down-and-out" and B>K:
        raise NotImplementedError("Error : Not Implemented")
    else:
        raise NotImplementedError("Error of type barriere name")
    return price
    


# -------------------------------------------------------------
# Biais d'observation discret
# -------------------------------------------------------------
def etude_biais_monitoring(S, K, r, q, sigma, T, B,
                           type_barriere="down-and-in", kind="put",
                           liste_pas=(4, 12, 52, 252, 1000),
                           n_sims=200_000, seed=None):
    prix = []
    liste_biais = []
    errs = []
    prix_continu = prix_barriere_ferme(S, K, r, q, sigma, T, B, type_barriere=type_barriere, kind=kind)
    for n in liste_pas:
        discret, err = prix_barriere_mc(S, K, r, q, sigma, T, B, type_barriere=type_barriere,
                     kind=kind, n_pas=n, n_sims=n_sims, seed=seed)
        biais = discret - prix_continu
        prix.append(discret)
        liste_biais.append(biais)
        errs.append(err)
    prix = np.asarray(prix)
    liste_biais = np.asarray(liste_biais)
    errs = np.asarray(errs)
    return {"n_pas":np.asarray(liste_pas),"prix":prix,"err_std":errs,"biais":liste_biais,"reference":prix_continu}
    


# -------------------------------------------------------------
# Correction de la barrière avec Broadie-Glasserman-Kou
# -------------------------------------------------------------
def correction_broadie(B, sigma, T, n_pas, sens="down"):
    beta= 0.5826 # = -zeta(1/2) / sqrt(2pi)
    if sens=="down":
        B_corrigee = B*np.exp(+beta*sigma*np.sqrt(T/n_pas))
    elif sens=="up":
        B_corrigee = B*np.exp(-beta*sigma*np.sqrt(T/n_pas))
    else:
        raise ValueError("please enter 'up' or 'down'")
    return B_corrigee
    


if __name__ == "__main__":
    S0, K0, r0, q0, v0, T0 = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0
    B0 = 70.0
    print(f"put vanille = {bs.price(S0, K0, r0, q0, v0, T0, 'put'):.6f}")
    print(f"DI ferme    = {prix_barriere_ferme(S0, K0, r0, q0, v0, T0, B0):.6f}")
    print(f"DI MC       = {prix_barriere_mc(S0, K0, r0, q0, v0, T0, B0, seed=42)}")
