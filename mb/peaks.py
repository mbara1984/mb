from scipy.optimize import curve_fit
import numpy as np
from importlib import reload 

import  scipy.optimize as opt 
reload(opt)

def gaussian_peak(x,*params):
    y = np.zeros_like(x+0.0)
    #y += params[0]*0 # offset
    ctr = params[0]
    amp = params[1]
    wid = params[2]    
    y = y + amp * np.exp( -((x - ctr)/wid)**2) # GAUSSIAN PEAKS
    return y

def gaussLorentzApproxPeak(x,*params):

    y = np.zeros_like(x+0.0)
    #y0  = params[0]*0 # offset
    ctr = params[0]
    amp = params[1]
    wid = params[2]
    rho = params[3] # weight gaussian to lorentzian shape
    #rho=1
    y =  amp*( rho * np.exp( -4*np.log(2)*( (x - ctr)/wid)**2) + (1.-rho)/ ( 1+(2*(x-ctr)/wid)**2 ))
    return y

    
def fit_peak_around(x,y,x0=5,sig=1,peak="lg", maxfev = 14600, xtol=1e-14):

    if peak == "lg":
        func = gaussLorentzApproxPeak
    #    guess = [0,x0,max(y),sig,0]
    else: # gaussian peak
        func = gaussian_peak
    guess = [x0,max(y),sig,1]
    lb = [0.0,0,np.minimum(sig/3,.1),0]
    ub = [len(x),2*max(y),sig*13,1]
    
    bounds = [lb,ub]
    popt, pcov = opt.curve_fit(func, x, y, p0=guess,  bounds=bounds,maxfev = maxfev,  xtol=xtol)
    #popt = guess
    #print(guess)    
    #print(popt)
    fit = func(x, *popt)
    return popt[0],popt[1],popt[2],popt[3],fit # x0,amp,fwhm

def fit_peak_by_peak(yy0,n=3,sigma=3, maxfev = 14600, xtol=1e-14):
    xx = np.arange(len(yy0))+0.0
    yy=yy0.copy()
    posL = []
    fwhmL = []
    ampL = []
    rhoL = []

    tfit = yy*0
    x00=0

    for i in range(n):
        #print(i,yy.max(),len(yy==yy.max()),len(xx))
        #print(xx[yy==yy.max()][0])
        #x0 ,amp, std, fit = cali.fit_my_peaks(xx,yy,3)
        x00 = xx[yy==yy.max()][0]
        x0, amp ,dx0, rho, fit = fit_peak_around(xx,yy,x00,sig=sigma, maxfev = maxfev,  xtol=xtol)
        posL.append(x0)
        ampL.append(amp)
        fwhmL.append(dx0)
        rhoL.append(rho)
        
        yy-=fit
        tfit+=fit
    return np.array(ampL),np.array(posL),np.array(fwhmL),np.array(rhoL),tfit,((tfit-yy0)**2).sum()

def fit_one_shot(yy0,n=3,maxfev = 14600, xtol=1e-14):
    import scipy.optimize as opt 
    #reload(opt)
    yy=yy0.copy()
    def func(x,*params):
        #print("params len",len(params))
        y = gaussLorentzApproxPeak(x,*params[0:4])+0.0
        for k in range(1,n):
            y= y+ gaussLorentzApproxPeak(x,*params[(k*4):(k*4+4) ])
        return y

    x = np.arange(len(yy0))+0.0

    amp, pos, fwhm , rho, fitP,er = fit_peak_by_peak(yy,n=n, maxfev = maxfev,  xtol=xtol)

    #print("Pre fit done:")
    #print(amp, pos, fwhm , rho)
    geuss = []
    for k in range(n):
        geuss = geuss + [pos[k],amp[k],fwhm[k],rho[k]]        
    #    ctr = params[0]
    #    amp = params[1]
    #    wid = params[2]
    #    rho = params[3] # weight gaussian to lorentzian shape
    fitg = func(x, *geuss)

    lb = [0,0,.0001,-1111]*n
    ub = [len(x),11.1*max(yy0)+0.01,len(x),1]*n
    bounds = [lb,ub]
    #print(np.array(lb).reshape(-1,4))
    #print(np.array(ub).reshape(-1,4))
    
    def in_bounds(x, lb, ub):
        """Check if a point lies within bounds."""
        return np.all((x >= lb) & (x <= ub))
    #geuss[-1]=111111111111    
    geuss = np.array(geuss)
    ddx = np.array((False == (np.array((geuss >= lb) * (geuss <= ub)))))

    #print(geuss,ddx[::2])
    #print("ddx "*7,ddx)
    #print(in_bounds(geuss, lb, ub))
    midx=np.arange(len(geuss))
    lbub = np.hstack([lb,ub]).ravel()
    if ddx.sum()>0:
        geuss[midx[ddx]] =  lbub[midx[ddx]]
        print('init err')
    #ddx = False == (np.array((geuss >= lb) * (geuss <= ub)))    
    #print(ddx)

    # print(geuss)
    # print(np.array(geuss).reshape(-1,4))
    
    # print((geuss >= lb) * (geuss <= ub))

    # print("x0")
    # print(geuss[ddx])
    # print("lower")
    # print(np.array((lb))[ddx])
    # print("upper")
    # print(np.array(ub)[ddx])
    # print(ub)
    #popt, pcov = opt.curve_fit(func, x, yy0, p0=guess,  maxfev = 13*4600, bounds = bounds,method = 'lm', 'trf', 'dogbox'})
    popt, pcov = opt.curve_fit(func, x, yy, p0=geuss,  bounds = bounds, maxfev = maxfev,  xtol=xtol)    
    fit = func(x, *popt)

    amps  = []
    fwhm  = []
    pos   = []
    rho   = []
    fs=[]
    for k in range(n):
        a,b,c,d = popt[(k*4):(k*4+4)]
        fs.append(gaussLorentzApproxPeak(x,*popt[(k*4):(k*4+4) ]))
        pos.append(a)
        amps.append(b)
        fwhm.append(c)
        rho.append(d)

    ii = np.argsort(amps)[::-1]
    # x0,amp,fwhm,rho,fs,fit
    return np.array(pos)[ii],np.array(amps)[ii],np.array(fwhm)[ii], np.array(rho)[ii], np.array(fs)[ii],fit,pcov

def test_peaks():

 
    #from generators import generate_measurements
    #from models import create_ramnet2c_model, save_ramnet2b_model

    import numpy as np

    n = 1024
    m = 2048*8
    #m = 2048*2

    EPOCHS  = 32 #1*64//2#//8
    num_ep  =  EPOCHS
    INIT_LR =  0.01 #1e-3
    BATCH_SIZE = 32
    VAL_SPLIT  = 0.25
    do_weight_init = False
    num_batches_per_epoch = (1.0-VAL_SPLIT)*m/BATCH_SIZE


    from utils import generators as gene

    noise =0.005
    photon_count=1e7
    baseline_level=1
    seed=12
    fixpeaks = np.array([12,55,211,144,100])*1024/256
    [measure_train,
     bl_train,
     ramanI_train,
     ramanF_train,
     conc_train,
     raman_signal_train] = gene.generate_measurements(n=n, m=m,
                                                 peaks=fixpeaks,
                                                 amps=[1.2,.55,.11,.88,1.05],
                                                 fwhm=[1.2,2,3,4,7],
                                                 npeaks=10,
                                                 min_conc=0.0,
                                                 max_conc=0.5,
                                                 photon_count=photon_count,
                                                 noise_level=noise,
                                                 seed=seed,
                                                 randomize_n=True,
                                                 baseline_level=baseline_level )

    #####

    amp, xx0, fwhm, fit =     fit_peak_by_peak(raman_signal_train[0,:],n=6) 
    plot
