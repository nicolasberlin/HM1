import time
import matplotlib.pyplot as plt
import numpy as np, time


__package__= "utils"

def phi(z: np.ndarray):
    """
    Smoothed hinge loss (phi) et sa dérivée (phi').
    Entrée:
      z: array numpy (1D/2D ok)
    Sorties:
      u = phi(z), v = phi'(z)  (même shape que z)
    """

    # S'assurer que z est un tableau numpy
    z = np.asarray(z, dtype=np.float64)
    u = np.zeros_like(z)  
    v = np.zeros_like(z)   


    # Définir les masques pour chaque région
    M1 = z <= -1                # Région 1 : z <= -1 (perte nulle)
    M2 = (z > -1) & (z < 0)     # Région 2 : -1 < z < 0 (zone de transition lissée)
    M3 = z >= 0                 # Région 3 : z >= 0 (perte linéaire)

    # Region 1 : z <= -1 already initialized to zero

    # Region 2 : -1 < z < 0
    u[M2] = 0.5 * (1 + z[M2])**2   
    v[M2] = 1 + z[M2]              

    # Region 3 : z >= 0
    u[M3] = 0.5 + z[M3]           
    v[M3] = 1.0                    

    # Return u = phi(z) and v = phi'(z)
    return u, v 



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
    print("min distance to {−1,0} =", float(delta.min()))

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


def sigma_max_sq(X):
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError("X contient NaN/Inf")
    s = np.linalg.svd(X, compute_uv=False)
    return float(s[0]**2)


def gradient_descent(train, lam=5e-3, max_iter=200000, tol=1e-9,
                     rng=0, theta0=None, time_budget_sec=180, eps=1e-3):
    """
    Gradient descent for SVM with smoothed hinge loss and L2 regularization.
    Parameters:
      - train: train data with X(d, m) data matrix and 'y' (m,)
      - lam: float regularization strength (lambda)
      - max_iter: int maximum number of iterations
      - tol: float tolerance for gradient norm stopping criterion
      - rng: int random seed for theta0 if theta0 is None
      - theta0: (d,) initial parameter vector (if None, random init)
      - time_budget_sec: float maximum time budget in seconds
      - eps: float relative tolerance for gradient norm stopping criterion
    Returns:
      - theta: (d,) estimated parameter vector
      - info: dict with keys:
          'f': list of objective values per iteration
          'gnorm': list of gradient norms per iteration
          'time': list of elapsed times per iteration
          'alpha': step size used
          'L': Lipschitz constant used
    """

    X = train["X"][0,0]; y = train["y"][0,0].ravel()
    d = X.shape[0]

    L = sigma_max_sq(X) + lam
    alpha = 1.0 / L

    gen = np.random.default_rng(rng)
    theta = theta0 if theta0 is not None else gen.standard_normal(d)

    f_hist, gnorm_hist, t_hist = [], [], []
    t0 = time.perf_counter()

    for k in range(max_iter):
        f, g = svm(train, theta, lam)
        gnorm = float(np.linalg.norm(g))
        now = time.perf_counter() - t0

        f_hist.append(float(f)); gnorm_hist.append(gnorm); t_hist.append(now)

        if k == 0:
            g0 = gnorm  # norme initiale

        # arrêts
        if gnorm <= tol: break
        if gnorm / g0 <= eps: break
        if now >= time_budget_sec: break

        theta = theta - alpha * g

    return theta, {"f": f_hist, "gnorm": gnorm_hist, "time": t_hist,
                   "alpha": alpha, "L": L}
