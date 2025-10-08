import time
import numpy as np
from .svm import svm

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

