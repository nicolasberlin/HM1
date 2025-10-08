import time
import numpy as np
import matplotlib.pyplot as plt
from .svm import svm
from .gradient_descent import sigma_max_sq


__all__ = [
    "linesearch",
    "ls_gradient_descent",
    "showMNISTImage",
    "showMNISTImages_many",
    "binary_classifier_accuracy",
]

def linesearch(train,v,lam=5e-3,theta0=None,alphabar= 10,c=1e-4,rho=0.5,L=None):
  
    X = train["X"][0,0]; y = train["y"][0,0].ravel()
    d = X.shape[0]
    theta = theta0 
    if L is None:
        L = sigma_max_sq(X) + lam
    alphamin = 1.0 / L
 
    alpha = alphabar
    f,gradient =svm(train, theta, lam)
    theta_new = theta + alpha * v
    fnextval, _ = svm(train, theta_new, lam)
    t=c* np.dot(gradient,v)

    while alpha>=alphamin and fnextval > f +t* alpha :
        alpha= rho*alpha
        theta_new = theta + alpha * v
        fnextval,_=svm(train, theta_new, lam)
    if rho*alpha< alphamin : print('Backtracking minimum step size reached!')
    return alpha




def ls_gradient_descent(train, lam=5e-3, max_iter=30000, tol=1e-9,
                     rng=0, theta0=None, time_budget_sec=180, eps=1e-3,alphabar= 8,c=1e-4,rho=0.5):
 

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
        alpha=linesearch(train,v=-g,lam=lam,theta0=theta,alphabar= alphabar,c=c,rho=rho)
        theta = theta - alpha * g
        if k == 0:
            g0 = gnorm  

        # arrêts
        if gnorm <= tol: break
        if gnorm / g0 <= eps: break
        if now >= time_budget_sec: break
       
    print(f"Final iteration: {k}")
    print(f"Final gradient norm: {gnorm}")
    print(f"Relative gradient norm: {gnorm / g0}")
    return theta, {"f": f_hist, "gnorm": gnorm_hist, "time": t_hist,
                   "alpha": alpha, "L": L}




def showMNISTImage(I):
    if len(I)==785:
        I=I[:-1]
  
    if np.size(I)!= 784:
        print("Image must have length 784 (28x28)")


    img = I.reshape(28, 28) #reshape function from matlab code
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(img * 255, cmap='gray', aspect='equal') #to make a gray color map
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    return fig




def showMNISTImages_many(X):
    

    if X.shape[0] == 785:
        X = X[:-1, :]
    
    n_images = X.shape[1]
    k = int(np.ceil(np.sqrt(n_images))) 
    
    I = np.zeros((k * 28, k * 28))
    
    for k1 in range(k):
        for k2 in range(k):
            kk = (k1)* k + k2
            if kk >= n_images:
                break
            I[k1*28:(k1+1)*28, k2*28:(k2+1)*28] = X[:, kk].reshape(28, 28)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(I * 255, cmap='gray', aspect='equal') #to make a gray color map
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')
    return fig


def binary_classifier_accuracy(data, theta):
   
    X = data['X'][0][0]      
    y = data['y'][0][0]    
    
    z = X.T @ theta    
    
    ycond = (z > 0).astype(int)
    mistakes = np.where(y[:,0] != ycond )[0]  
    
    n_correct = len(y) - len(mistakes)
    accuracy = n_correct / len(y)
   
    
    return accuracy, mistakes