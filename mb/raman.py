import numpy as np
import os
from sklearn.decomposition import NMF
from scipy.optimize import minimize, linprog

def reorder_autoraman(ff):
    x=[]
    y=[]
    for f in ff:
        p = os.path.basename(f)
        x.append(int(p[0]))
        y.append(int(p[2]))
    x=np.array(x);y=np.array(y)
    ii=np.argsort(x)
    f2 = np.array(ff)[ii]#
    x=x[ii];y=y[ii]
    for kk in np.unique(x):
        ee=np.argsort(y[x==kk])
        f2[x==kk] = f2[x==kk][ee]
    return f2


def LieberFit(S,order=5,tot_iter=55):
    """
    % Lieber fit
    % S is a matrix of input spectra with each observation in the rows
    % order is the order of polynomial fit
    % tot_iter is the number of iterations for the code to fit the baseline
    function outspectrum = lieberfit(S, order, tot_iter)
    """
    x = np.linspace(0,1,len(S))
    polyspec_iter = S+0.0
    out=[]
    for i in range(tot_iter):
        p_order = np.polyfit(x,polyspec_iter,order)
        polyspec_order=np.polyval(p_order,x)
        polyspec_iter = np.minimum(polyspec_order, polyspec_iter)
    out = S - polyspec_iter
    return out, polyspec_iter


def specrum_NMF_decomposition(hsi,NC=5,prefilter=False,postfilter=False):
    #hsi[~np.isfinite(hsi)]=0
    from sklearn.decomposition import PCA as sk_pca
    from sklearn.decomposition import FastICA
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import StandardScaler
    from sklearn import svm
    from sklearn.cluster import KMeans
    from scipy.signal import savgol_filter
    
    if prefilter:
        hsi1 = savgol_filter(hsi, 55, polyorder = 3,deriv=0,axis=1)
        nfeat0 = hsi1
    else:
        nfeat0 = hsi
    print(nfeat0.shape,nfeat0.max())
    nmf_ = NMF(n_components=NC, init='nndsvd', random_state=0,max_iter=4500)
    xnmf = nmf_.fit_transform(nfeat0)  # Reconstruct signals
    ee = nmf_.transform(nfeat0)
    re=nmf_.inverse_transform(ee)

    #figure()
    #plot(re.T);title('reco');
    #figure()
    #for k in range(NC):
    #   plot(nmf_.components_[k])

    if postfilter:
        nmf_.components_ = savgol_filter(nmf_.components_, 55, polyorder = 3,deriv=0,axis=1)
    
    return nmf_.components_, nmf_


def extrac_known_tBG(aa,w):
    """ Wbg ba dectionary (not  orthogonal, mostly NMF-based )"""
    # Wbg = np.load("/home/mbara/work/CAMP/raman/data/reference_data/raman_glass_bgNov2020.npy")
    # w,nmf = spec_NMF(bb,4,prefilter=True) 
    phi = lambda x: ((aa-x.dot(w) )**2).sum()
    cons = [{"type": "ineq", "fun": lambda x: aa-(x @ w) }]# a > x @ w
    re = minimize(phi,x0=np.random.rand(w.shape[0])*100,constraints=cons) #bounds=[[-10,None]]*4
    print(re.message)
    return re.x, aa-re.x.dot(w)


def linear_reg(x,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x, y)
    f = model.intercept_+ model.coef_*x
    
    return model.intercept_, model.coef_, r_sq,f # ax+b
