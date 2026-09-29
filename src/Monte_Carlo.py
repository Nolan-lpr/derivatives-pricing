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

#rmq : on pourrait combiner les 2 méthodes ici


#Valeurs tests d'affichage
if __name__ == "__main__":
    S0, K0, r0, q0, v0, T0 = 100.0, 100.0, 0.03, 0.01, 0.20, 1.0
    ref = bs.price(S0, K0, r0, q0, v0, T0, "call")
    print(f"reference Black-Scholes = {ref:.6f}")
    print(f"Monte Carlo             = {prix_mc(S0, K0, r0, q0, v0, T0, 'call', seed=42)}")



#On construit maintenant les Greeks, avec Monte-Carlo (par différence finies), car BS ne sera pas applicable à notre projet final d'autocall


#DELTA
def delta_mc(S, K, r, q, sigma, T,h, kind="call", n_sims=100_000, seed=0,seed2=None):
    S_plus = simuler_terminal(S+h, r, q, sigma, T, n_sims, seed=seed)
    S_moins = simuler_terminal(S-h, r, q, sigma, T, n_sims, seed=seed if seed2==None else seed2) #2 seed pour les tests du notebook sur l'influance des titages indépendents
    payoff_plus = _payoff(S_plus,K,kind=kind)
    payoff_moins = _payoff(S_moins,K,kind=kind)
    diff = payoff_plus-payoff_moins
    diff_act = diff*np.exp(-r*T)
    tableau_delta_estim = diff_act/(2*h)
    delta_estim = np.mean(tableau_delta_estim)
    erreur_std = np.std(tableau_delta_estim,ddof=1)/np.sqrt(n_sims)
    return (delta_estim,erreur_std)

#GAMMA
def gamma_mc(S, K, r, q, sigma, T,h, kind="call", n_sims=100_000, seed=0):
    S_plus = simuler_terminal(S+h, r, q, sigma, T, n_sims, seed=seed)
    S_moins = simuler_terminal(S-h, r, q, sigma, T, n_sims, seed=seed)
    S_central = simuler_terminal(S, r, q, sigma, T, n_sims, seed=seed)
    payoff_plus = _payoff(S_plus,K,kind=kind)
    payoff_moins = _payoff(S_moins,K,kind=kind)
    payoff_central = _payoff(S_central,K,kind=kind)
    diff = payoff_plus-2*payoff_central+payoff_moins
    diff_act = diff*np.exp(-r*T)
    tableau_gamma_estim = diff_act/(h**2)
    gamma_estim = np.mean(tableau_gamma_estim)
    erreur_std = np.std(tableau_gamma_estim,ddof=1)/np.sqrt(n_sims)
    return (gamma_estim,erreur_std)
    
#VEGA
def vega_mc(S, K, r, q, sigma, T,h, kind="call", n_sims=100_000, seed=0):
    Vol_plus = simuler_terminal(S, r, q, sigma+h, T, n_sims, seed=seed)
    Vol_moins = simuler_terminal(S-h, r, q, sigma-h, T, n_sims, seed=seed)
    payoff_plus = _payoff(Vol_plus,K,kind=kind)
    payoff_moins = _payoff(Vol_moins,K,kind=kind)
    diff = payoff_plus-payoff_moins
    diff_act = diff*np.exp(-r*T)
    tableau_vega_estim = diff_act/(2*h)
    vega_estim = np.mean(tableau_vega_estim)
    erreur_std = np.std(tableau_vega_estim,ddof=1)/np.sqrt(n_sims)
    return (vega_estim,erreur_std)

#RHO
def rho_mc(S, K, r, q, sigma, T,h, kind="call", n_sims=100_000, seed=0):
    rho_plus = simuler_terminal(S, r+h, q, sigma, T, n_sims, seed=seed)
    rho_moins = simuler_terminal(S, r-h, q, sigma, T, n_sims, seed=seed)
    payoff_plus = _payoff(rho_plus,K,kind=kind)
    payoff_moins = _payoff(rho_moins,K,kind=kind)
    payoff_act_plus = payoff_plus*np.exp(-(r+h)*T)
    payoff_act_moins = payoff_moins*np.exp(-(r-h)*T)
    diff_act = payoff_act_plus-payoff_act_moins
    tableau_rho_estim = diff_act/(2*h)
    rho_estim = np.mean(tableau_rho_estim)
    erreur_std = np.std(tableau_rho_estim,ddof=1)/np.sqrt(n_sims)
    return (rho_estim,erreur_std)    

def theta_mc(S, K, r, q, sigma, T,h, kind="call", n_sims=100_000, seed=0):
    theta_plus = simuler_terminal(S, r, q, sigma, T+h, n_sims, seed=seed)
    theta_moins = simuler_terminal(S, r, q, sigma, T-h, n_sims, seed=seed)
    payoff_plus = _payoff(theta_plus,K,kind=kind)
    payoff_moins = _payoff(theta_moins,K,kind=kind)
    payoff_act_plus = payoff_plus*np.exp(-r*(T+h))
    payoff_act_moins = payoff_moins*np.exp(-r*(T-h))
    diff_act = payoff_act_plus-payoff_act_moins
    tableau_theta_estim = -diff_act/(2*h*bs.DAYS_PER_YEAR)
    theta_estim = np.mean(tableau_theta_estim)
    erreur_std = np.std(tableau_theta_estim,ddof=1)/np.sqrt(n_sims)
    return (theta_estim,erreur_std)    

