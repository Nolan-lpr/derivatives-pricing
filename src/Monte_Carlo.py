'Monte Carlo pricing of European vanilla options under Black-Scholes dynamics.'

import numpy as np
import black_scholes as bs

#Pour une option européenne, l'équation des mouvemment browniens se résous par : 
# S_T = S_0 * exp( (r - q - sigma^2 / 2) T  +  sigma sqrt(T) Z ) avec Z ~ N(0, 1)"

#On commence par simuler les chemins aléatoires
def simuler_terminal(S, r, q, sigma, T, n_sims, seed=None, antithetique=False):
    rng = np.random.default_rng(seed)
    drift = (r-q-sigma**2/2)*T
    vol = sigma*np.sqrt(T)
    if antithetique:
        n_paires = n_sims // 2
        Z = rng.standard_normal(n_paires)
        Z_moins = -Z
        Z = np.concatenate([Z,Z_moins])
    else:
        Z = rng.standard_normal(n_sims)

    S_T= S*np.exp(drift+vol*Z)
    return S_T


#Fonction pour calculer le payoff
def _payoff(S_T,K,kind):
    if kind=='call':
        payoff = np.maximum(S_T-K,0)
    elif kind=='put':
        payoff = np.maximum(K-S_T,0)
    else:
        raise ValueError("Kind Name Error")
    return payoff


#Pour pricer : on simule, on actualise, et on moyenne
def prix_mc(S, K, r, q, sigma, T, kind="call", n_sims=100_000, seed=None):
    S_T = simuler_terminal(S, r, q, sigma, T, n_sims, seed=seed)
    payoffs_actualise = _payoff(S_T,K,kind=kind)*np.exp(-r*T)
    moyenne = np.mean(payoffs_actualise)
    erreur_standard = np.std(payoffs_actualise,ddof=1)/np.sqrt(n_sims)
    return (moyenne,erreur_standard)


#On compare Monte-Carlo avec Black-Scholes
def etude_convergence(S, K, r, q, sigma, T, kind="call",
                      tailles=(10**3, 10**4, 10**5, 10**6), seed=None):
    prix = []
    erreurs_standard = []
    erreurs_reelles = []
    reference = bs.price(S,K,r,q,sigma,T,kind=kind)
    for i,n in enumerate(tailles):
        graine = None if seed is None else seed+i
        moyenne,erreur_standard = prix_mc(S, K, r, q, sigma, T, kind=kind, n_sims=n, seed=graine)
        prix.append(moyenne)
        erreurs_standard.append(erreur_standard)
        erreur_reelle = abs(moyenne-reference) #comparaison à Black_Scholes
        erreurs_reelles.append(erreur_reelle)
    return {'tailles':np.asarray(tailles),'prix':np.asarray(prix),'err_std':np.asarray(erreurs_standard),'erreur_reelle':np.asarray(erreurs_reelles),'reference':reference}


#Réduction de variance par la méthode des variables antithétiques (+Z et -Z de vol, moyenne de la somme)
def prix_mc_antithetique(S, K, r, q, sigma, T, kind="call",
                         n_sims=100_000, seed=None):
    S_T = simuler_terminal(S, r, q, sigma, T, n_sims=n_sims, seed=seed, antithetique=True)
    n_paires = n_sims//2
    S_plus = S_T[:n_paires]
    S_moins = S_T[n_paires:]
    moyenne_paires = 0.5*(_payoff(S_plus,K,kind=kind)+_payoff(S_moins,K,kind=kind))
    payoff_actualise = moyenne_paires*np.exp(-r*T)
    mean_payoff_actualise = np.mean(payoff_actualise)
    erreur_standard = np.std(payoff_actualise, ddof=1)/np.sqrt(n_paires)
    return (mean_payoff_actualise,erreur_standard)


#Réduction de variance par la méthode de la variable de contrôle
#On sait que la quantité simulée S_T à pour espérance S * exp((r - q) T)
# On corrige ensuite l'estimateur = payoff_moyen - beta * ( moyenne(S_T) - E[S_T] )
def prix_mc_variable_controle(S, K, r, q, sigma, T, kind="call",
                              n_sims=100_000, seed=None):
    S_T = simuler_terminal(S, r, q, sigma, T, n_sims, seed=seed)
    payoffs = _payoff(S_T,K,kind=kind)
    beta = np.cov(S_T,payoffs,ddof=1)[0,1]/np.var(S_T,ddof=1)
    estimateur = payoffs - beta*(S_T-S*np.exp((r-q)*T))
    estimateur_act = np.exp(-r*T)*estimateur
    moy_estimateur_act = np.mean(estimateur_act)
    erreur_standard = np.std(estimateur_act,ddof=1)/np.sqrt(n_sims)
    return (moy_estimateur_act,erreur_standard)


#Valeurs tests d'affichage
if __name__ == "__main__":
    S0, K0, r0, q0, v0, T0 = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0
    ref = bs.price(S0, K0, r0, q0, v0, T0, "call")
    print(f"reference Black-Scholes = {ref:.6f}")
    print(f"Monte Carlo             = {prix_mc(S0, K0, r0, q0, v0, T0, 'call', seed=42)}")