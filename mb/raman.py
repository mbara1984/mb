import numpy as np
import os
from sklearn.decomposition import NMF
from scipy.optimize import minimize, linprog
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter,  median_filter


def find_andor_header_length(f):
    ff=open(f)
    for k in range(50):
        s=ff.readline()
        if s=='\n':
            return k+1
    
def get_ramans(ff,header=None):
    a=[]
    if header is None:
        header=find_andor_header_length(ff[0])
    for f in ff:
        ps1 = np.genfromtxt(f,skip_header=header,delimiter=',')[:,1]
        rrrs = np.genfromtxt(f,skip_header=header,delimiter=',')[0:1024,0]
        ps1 = ps1.reshape(ps1.shape[0]//1024,1024).T
        ps1 = median_filter(ps1,(1,5)) # simplistic coosmic removal 
        nrm = ps1.sum(0)[np.newaxis,:]
        nrmMx=nrm.max()
        ps1 = ps1/nrm * nrmMx
        a.append(ps1.mean(1))
    return np.array(a),rrrs




def reorder_autoraman(ff):
    """ reaodrer filenames of autoraman.py """
    x=[]
    y=[]
    for f in ff:
        p = os.path.basename(f)
        x.append(int(p[-7]))
        y.append(int(p[-5]))
    x=np.array(x);y=np.array(y)
    ii=np.argsort(x)
    f2 = np.array(ff)[ii]#
    x=x[ii];y=y[ii]
    for kk in np.unique(x):
        ee=np.argsort(y[x==kk])
        f2[x==kk] = f2[x==kk][ee]
    return f2


def diff_bg_subtract(s1):
    dds1 = savgol_filter(s1, 7, polyorder = 3,deriv=1, axis=0,mode='nearest')
    w=dediff(dds1)
    w=w-w.min()

    return w


def dediff(dd):
    #dds1 = savgol_filter(s1, 13, polyorder = 2,deriv=1, axis=0)
    f0 = 0
    ff=[0]
    for k in range(1,len(dd)):
        ff.append(ff[-1]+dd[k])
    ff=np.array(ff)
    return ff#-ff.min()

def diffdiff_bg_subtract(s1): # verify this
    dds1 = savgol_filter(s1, 7, polyorder = 3,deriv=2, axis=0,mode='nearest')
    w=dediff(dediff(dds1))
    # remove  linear part 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(np.arange(len(s1)).reshape(-1,1),w.reshape(-1,1))
    w = -(model.intercept_[0]+np.arange(len(s1))*model.coef_[0]-w)
    w=w-w.min()
    return w







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
    x=np.array(x)
    y=np.array(y)    
    if x.ndim==1:
        x=x[:,np.newaxis]
        y=y[:,np.newaxis]        
    model.fit(x,y)
    r_sq = model.score(x, y)
    f = model.intercept_+ model.coef_*x
    
    return model.intercept_, model.coef_, r_sq,f # ax+b


def linear_reg_lod(x,y):
    from pylab import plot,legend
    ab, cov = np.polyfit(x,y,1, cov=True)# linear fit 
    sigma_a=np.sqrt(cov[0,0])
    sigma_b=np.sqrt(cov[1,1])
    fit=ab[0]*x+ab[1]
    LOD = 3*sigma_b/ab[0]
    #plot(x,fit,label='$y=a\cdot x+b$ \n$  a=%.2f\pm %.2f$\n$ b=%.1f\pm %.1f$' %(ab[0],sigma_a,ab[1],sigma_b))
    plot(x,y,'o')
    plot(x,fit,lw=3,alpha=.75,label='$y=a\cdot x+b$ \n$  a=%.3f\pm %.3f~~(rel~%.1f$%%$)$ \n$ b=%.3f\pm %.3f$\nLOD$_{3\\sigma}^{intercept}\\approx %.5f$' %(ab[0],sigma_a,abs(sigma_a/ab[0]*100),ab[1],sigma_b,LOD))    
    legend()
    return LOD


# def iso_lod(x,y):
#     K=1
#     I=len(x)
    
#     ab, cov = np.polyfit(x,y,1, cov=True)# linear fit 
#     sigma_a=np.sqrt(cov[0,0])
#     sigma_b=np.sqrt(cov[1,1])
#     fit=ab[0]*x+ab[1]
    
#     sigma_sq = (y-fit)**2/(I-2)
#     xhat  = (fit-ab[0])/ab[1]
#     wx_sq = 1/K +1/I + (x- xhat)**2/((x-xhat)**2).sum()

#     x0    = (0-ab[0])/ab[1]
#     w0_sq = 1/K +1/I + (0-x0)**2/((x-xhat)**2).sum()

#     delta = (y-fit)/np.sqrt(w0_sq*sigma_sq)

#     lod = delta * np.sqrt(w0_sq*sigma_sq)/ab[0]
#     return lod


def pls_x(x,y,n_components=10,prefilter=None):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import cross_val_predict
    
    import pylab as plt
    pls = PLSRegression(n_components=n_components)
    if prefilter is not None:
        x = prefilter(x)
    pls.fit(x , y) # fit spectra to predict 5 components concentrations

    #fe=linear_reg(y,Y_pred)
    #plot(y,Y_pred,'.'); plot(y,fe[-1],label='R$^2=%.4f$' % fe[-2]);legend()
    
    y_c = pls.predict(x)
    # Cross-validation
    y_cv = cross_val_predict(pls, x, y, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c  = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    print('R2 calib: %5.3f'  % score_c)
    print('R2 CV: %5.3f'  % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
    (score_c , score_c , mse_c,mse_cv)

    if prefilter is not None:
        pls.predict_filt = lambda xx: pls.predict(prefilter(xx))

    return pls, (mse_cv,score_cv)
