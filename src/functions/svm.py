import numpy as np
import matplotlib.pyplot as plt
from .loss import phi

def svm(train, theta, lam):
    """
    SVM objective with smoothed hinge loss and L2 regularization.

    Parameters:
      - train: train data with X(d, m) data matrix and 'y' (m,) labels (0/1)
      - theta: (d,) parameter vector
      - lam: float regularization strength (lambda)

    Returns:
      - f: float objective value
      - g: (d,) gradient with respect to theta
        """
    #print(f"is nam theta : ${np.any(np.isnan(theta))}")

    X = train["X"][0,0].astype(np.float64)
    y = train["y"][0,0].ravel().astype(np.float64)

    s = X.T @ theta

    delta = np.minimum(np.abs(s + 1.0), np.abs(s - 0.0))
    #print("min distance to {−1,0} =", float(delta.min()))

    loss_terms = y * phi(-s)[0] + (1 - y) * phi(s)[0]

    # valeur de la fonction objectif (perte + régularisation)
    f = loss_terms.sum() + (lam/2)*np.sum(theta**2)

    coeffs = -y * phi(-s)[1] + (1 - y) * phi(s)[1]   # (m,)
    
    g = X @ coeffs  + lam * theta
   
    return f, g


def check_gradient(train, lam=1e-3, rng=0):
    """
    Vérifie la justesse du gradient calculé par svm() via une approximation finie.
    Affiche un graphique log-log de l'erreur et estime la pente.
    """

    X = train["X"][0,0].astype(np.float64)
    y = train["y"][0,0].ravel().astype(np.float64)

    gen = np.random.default_rng(rng)
    theta = gen.standard_normal(X.shape[0]).astype(np.float64)

    # direction v unitaire pour la perturbation de theta
    v = gen.normal(size=X.shape[0])
    v /= np.linalg.norm(v)

    # Valeurs arbitraires pour faciliter le debug
    f0, g0 = svm(train, theta, lam)

    t = np.logspace(-8, 0, 101)
    error = np.empty_like(t)
    for i, ti in enumerate(t):
        theta_t = theta + ti*v
        try:
            f_t, _ = svm(train, theta_t, lam)
        except FloatingPointError as exc:
            print(
                f"Erreur numérique pour t={ti:.1e} : ||theta_t||={np.linalg.norm(theta_t):.3e}"
            )
            raise
        error[i] = abs(f_t - f0 - ti * np.dot(v, g0))    

    lo, hi = 30, 70
    slope = np.polyfit(np.log(t[lo:hi]), np.log(np.maximum(error[lo:hi], 1e-300)), 1)[0]
    print(f"Pente log-log estimée (zone centrale) ≈ {slope:.2f}")

    plt.figure(figsize=(6,4))
    plt.loglog(t, error, marker='o', linewidth=1)
    plt.xlabel("t"); plt.ylabel("|f(θ+tv)-f(θ)-t⟨v,∇f(θ)⟩|")
    plt.title("Gradient check (attendu: pente ≈ 2)")
    plt.grid(True, which="both"); plt.show()
