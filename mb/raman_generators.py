"""
Tools for generating raman spectra and simulating measurements
"""

def gaussian_peak(x,*params):
    y = np.zeros_like(x+0.0)
    #y += params[0]*0 # offset
    ctr = params[0] # center
    amp = params[1] # amplitude
    wid = params[2] # sigma   
    y = y + amp * np.exp( -((x - ctr)/wid)**2) # GAUSSIAN PEAKS
    return y

def gaussLorentzApproxPeak(x,*params):

    y = np.zeros_like(x+0.0)
    #y0  = params[0]*0 # offset
    ctr = params[0]
    amp = params[1]
    wid = params[2]
    rho = params[3] # weight gaussian to lorentzian shape 1= goussian
    
    y =  amp*( rho * np.exp( -4*np.log(2)*( (x - ctr)/wid)**2) + (1.-rho)/ ( 1+(2*(x-ctr)/wid)**2 ))
    return y


import numpy as np
import time
def spline_baseline(n=256,m=64,seed=None):
    """
    Generate spline-based baseline signal spectra
    
    Parameters
    ----------
    n : int
        Number of spectral channels in a single spectrum
    m : int
        Number of spectra to generate
    seed : int, optional
        Seed for the random number generator
    
    Returns
    -------
    out : ndarray
        Array with shape (m,n) containing the spectra
    """
    from scipy.interpolate import interp1d, BSpline

    # initialize random number generator
    # default argument of None automatically seeds with /dev/urandom or clock
    rs = np.random.RandomState(seed)

    # parameters for the B-spline
    order = 4
    control_points = 6#11 # number of basis functions
    
    # create knots for open uniform B-spline
    knots = np.concatenate((np.zeros(order-1),np.linspace(0,1,control_points+2-order),np.ones(order-1)))

    # create output coordinate system
    x_out = np.linspace(0,1,n)

    # generate random control points
    control_point_values = rs.rand(m, control_points)

    # evaluate B-splines
    return BSpline(knots, control_point_values, order-1, axis=1)(x_out) #+ (15+5*(np.random.rand()-0.5))

def generate_raman(n, m, peaks, amps, fwhm=None, min_fwhm=2, max_fwhm=12, fwhm_ig =0.5, seed=None, randomize_n=False, varfwhm=0.0, rho=None):
    """
    Generate Raman spectra

    Parameters
    ----------
    n : int
        Number of spectral channels in a single spectrum
    m : int
        Number of spectra to generate
    peaks : list
        Location of Raman peaks (values of None indicate random location)
    amps : list
        Amplitudes of Raman peaks (values of None indicate random amplitude)
    seed : int, optional
        Seed for the random number generator

    Returns
    -------
    out : ndarray
        Array with shape (m,n) containing the spectra
    """
    if fwhm is None:
        fwhms=[None]*len(peaks)
    else:
        fwhms=fwhm
    if rho is None:
        rhos=[1]*len(peaks) # def is rho = 1 so gaussian 
    else:
        rhos=rho
    # initialize random number generator
    # default argument of None automatically seeds with /dev/urandom or clock
    rs = np.random.RandomState(seed)

    # I. construct m-raman spectra (n-spectral channels)
    #    peaks position and amplitude defined + gaussian profile
    s = np.arange(n,dtype=np.float)
    s.shape = (1,n)

    raman = np.zeros((m,n))
    import scipy.stats as st       
    for (peak, amp, fwhm,rho) in zip(peaks, amps, fwhms, rhos):
        if peak is None:
            peak = rs.rand(m,1)*n
        if amp is None:
            amp = rs.rand(m,1)*0.5 # same as fixed concentration raman max amp
            if randomize_n:
                #  randomly kill half of peaks, results with peaks number in [0,npeaks] with mean = npeaks/2 
                amp[ rs.rand(m) < 0.5 ]=0 # gives kind of Gaussian peaks no. distribution around npeaks/2
        if fwhm is None:
            #fwhm = rs.rand(m,1)*(max_fwhm-min_fwhm)+min_fwhm
            fwhm = np.minimum(st.invgamma.rvs(fwhm_ig,size=(m,1)),10) # draw fwhm from inverse gamma distribution, limits value to 15
        else:
            #np.random.seed(np.uint32(1000*time.time()-1549948502.7539642 ))
            fwhm = fwhm*(1 + (np.random.rand(m,1)-.5)*varfwhm)
        #raman += amp*np.exp(-((s-peak)/fwhm)**2)
        raman +=  gaussLorentzApproxPeak(s,*[peak,amp,fwhm,rho])  #amp*np.exp(-((s-peak)/fwhm)**2)
        #ctr = params[0];    amp = params[1];    wid = params[2];    rho = params[3] # weight gaussian to lorentzian shape 1= goussian

    
    return raman

def random_raman(n=256, m=64, npeaks=7, min_fwhm=2, max_fwhm=12, seed=None,randomize_n=False):
    """
    Generate random Raman spectra

    Parameters
    ----------
    n : int
        Number of spectral channels in a single spectrum
    m : int
        Number of spectra to generate
    npeaks : int
        Number of Raman peaks to simulate
    seed : int, optional
        Seed for the random number generator

    Returns
    -------
    out : ndarray
        Array with shape (m,n) containing the spectra
    """
    return generate_raman(n,m,[None]*npeaks,[None]*npeaks,seed=seed, randomize_n = randomize_n)

def fixed_raman(n=256, m=64, peaks=[32,66,100,166,200], amps=[1,.3,1.2,.81,1.4],fwhm =[1,2,1.66,5,4 ], seed=None,rho=None):
    """
    Generate Raman spectra with fixed peaks

    Parameters
    ----------
    n : int
        Number of spectral channels in a single spectrum
    m : int
        Number of spectra to generate
    peaks : ndarray
        Location of Raman peaks
    amps : ndarray
        Amplitudes of Raman peaks
    seed : int, optional
        Seed for the random number generator

    Returns
    -------
    out : ndarray
        Array with shape (m,n) containing the spectra
    """
    return generate_raman(n,m,peaks,amps,fwhm,seed=seed,rho=rho)


def generate_measurements(n, m, peaks, amps, npeaks, fwhm,
                          min_conc=0.0,
                          max_conc=0.5,
                          photon_count=np.inf,
                          noise_level=0.001,
                          seed=None,
                          randomize_n=False,baseline_level=1.0,rho=None):
    """
    Generate a set of simulated Raman measurements

    Parameters
    ----------
    n : int
        Number of spectral channels in a single spectrum
    m : int
        Number of spectra to generate
    peaks : ndarray
        Location of target Raman peaks
    amps : ndarray
        Amplitudes of target Raman peaks
    npeaks : int
        Number of background Raman peaks to simulate
    min_conc : float, optional
        Minimum concentration value for target
    max_conc : float, optional
        Maximum concentration value for target
    photon_count : float, optional
        Number of photons per spectra, for Poisson noise
    noise_level : float, optional
        Gaussian noise level
    seed : int, optional
        Seed for the random number generator
    randomize_n:  Bool, optional
        if True npeaks is maximal number of raman background  peaks,
        number of peaks is randomly drawn from [0-npeaks]

    Returns (in order)
    ------------------
    measure : ndarray
        The noisy measurements, of size (m,n) 
    bl : ndarray
        The true baseline spectra, of size (m,n)
    ramanI : ndarray
        The true background Raman spectra, of size (m,n)
    ramanF : ndarray
        The true target Raman spectra, of size (m,n)
    conc : ndarray
        The true concentrations, of size (m,1)
    raman_signal : ndarray
        The true target Raman spectra with concentrations, of size (m,n)
    """

    # seed random number generator
    import time
    if seed is None:
        seed=int(time.time())
    rs = np.random.RandomState(seed)      

    # generate spectra
    print ("Creating baseline...")
    bl = spline_baseline(n=n, m=m, seed=seed)
    print ("Creating random background Raman spectra...")
    ramanI = random_raman(n=n, m=m, npeaks=npeaks, seed=seed+2, randomize_n=randomize_n)
    print ("Creating target Raman spectra...")
    print(fwhm)
    ramanF = fixed_raman(n=n, m=m, peaks=peaks, amps=amps, fwhm=fwhm, seed=seed,rho=rho)

    # total strength of raman signal (our unknown)
    conc = rs.rand(m,1)*(max_conc-min_conc)+min_conc

    # raman signal that we are actually interested in
    print ("Mixing spectra...")
    raman_signal = conc*ramanF
    measure = raman_signal + bl*baseline_level + ramanI

    if photon_count < np.inf and photon_count > 0:
        # Poisson noise
        print ("Applying Poisson noise...")
        total_vals = np.sum(measure, axis=1, keepdims=True)
        scale_factors = photon_count / total_vals
        photon_intensity = measure * scale_factors
        photons = rs.poisson(photon_intensity)
        measure = photons / scale_factors
    
    # Gaussian noise
    print ("Adding Gaussian noise...")
    measure+=np.random.randn(*measure.shape)*measure.max()*noise_level

    return measure, bl, ramanI, ramanF, conc, raman_signal
