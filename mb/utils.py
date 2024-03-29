## PLOTS AND GENERAL STUFF
import numpy as np    
def saveMyFig(fname='',figdir=''):
    if fname=='':
        import tempfile
        fname = tempfile.mktemp()
    if figdir!='':
        import os
        try:
            os.makedirs(figdir)
        except: # assuming dir exists ... ugly...
            pass
        fname = figdir+"/"+fname
    from pylab import savefig
    
    savefig(fname+".pdf")
    savefig(fname+".png")
    savefig(fname+".svg")

    return fname+'.pdf'

def plotyy(a,b):
    """
    ax1,ax2=plotyy((t,data1),(t,data2))

    ax1.plot(t,-data1,'r') #to plot on ax1 axis
    ax2.plot(t,-data2,'r') #to plot on ax2 axis     
    use:
    
    ax2.set_ylabel('sin', color=color)  
    etc
    """

    from matplotlib.pylab import plt 
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(*a,color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(*b, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    return ax1,ax2


def plotxx(a,b):
    """
    ax1,ax2=plotyy((t,data1),(t,data2))

    ax1.plot(t,-data1,'r') #to plot on ax1 axis
    ax2.plot(t,-data2,'r') #to plot on ax2 axis     
    use:
    
    ax2.set_ylabel('sin', color=color)  
    etc
    """
    from matplotlib.pylab import plt 
    
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.plot(*a,color=color)
    ax1.tick_params(axis='x', labelcolor=color)

    ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.plot(*b, color=color)
    ax2.tick_params(axis='x', labelcolor=color)

    return ax1,ax2


def plot_t(x,y,t=None,colormap='inferno'):

    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    if t is None: # so can see speed
        t = np.linspace(0,1,x.shape[0]) # your "time" variable

    # set up a list of (x,y) points
    points = np.array([x,y]).transpose().reshape(-1,1,2)


    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    #print segs.shape  # Out: ( len(x)-1, 2, 2 )
                      # see what we've done here -- we've mapped our (x,y)
                      # points to an array of segment start/end coordinates.
                      # segs[i,0,:] == segs[i-1,1,:]

    # make the collection of segments
    lc = LineCollection(segs, cmap=plt.get_cmap(colormap))
    lc.set_array(t) # color the segments by our parameter

    # plot the collection
    plt.gca().add_collection(lc) # add the collection to the plot
    plt.xlim(x.min(), x.max()) # line collections don't auto-scale the plot
    plt.ylim(y.min(), y.max())




def short_periodogram(a,fft_size=128,df=1,step=16):
    """ moving window periodogram """
    from scipy.signal import periodogram
    window = np.hanning(fft_size)
    res=[]
    for i in range(0,len(a)-fft_size,step):
        aa = a[i: (i+ fft_size)]
        p=periodogram(aa*window,df)
        res.append(p[1][1:])
    return np.array(res),2*np.pi*p[0]
    

def inset(plot_cmd, tit="",loc=[.15, .675, .2, .2]):
    a = axes(loc, axisbg='y')
    #    n, bins, patches = hist(s, 400, normed=1)
    exec(plot_cmd)
    title(tit)
    setp(a, xticks=[], yticks=[])
    return a

def mb_show(img):
    import pylab
    if type(img) != type([]):
        img=[img]
    n  = len(img) # # how many images
    nn = np.ceil(np.sqrt(n+0.0)).astype(np.int64)
    n1 =nn; m1=nn
    fig = pylab.figure()                                                 
    ax = fig.add_subplot(m1,n1,1)
    jj = 0
    for i in range(m1):
        for j in range(n1):
            ax = fig.add_subplot(m1,n1,jj+1)
            if jj<n:
                ax.imshow(img[jj])
            jj+=1

def animate_hsi(hsi):

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig = plt.figure()

    k = 0
    im = plt.imshow(hsi[:,:,k], vmax=153.0,animated=True)
    #im = plt.imshow(hsi[:,:,k], animated=True)
    
    def updatefig(*args):
        global k,hsi
        k+=1
        print(k)
        im.set_array(hsi[:,:,k])
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True)
    plt.show()


def animate_hsi_cv(hsi,fileout='/tmp/output',Nout=512):
    from scipy.interpolate import interp2d
    import numpy as np
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    #cap = cv2.VideoCapture(0)

    x = np.arange(hsi.shape[0])
    y = np.arange(hsi.shape[1])
    xn = np.linspace(0,hsi.shape[0],hsi.shape[0]*6)
    yn = np.linspace(0,hsi.shape[1],hsi.shape[1]*6)
    Nxout = xn.shape[0]
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('output.avi',fourcc, 20.0, [hsi.shape[0],hsi.shape[1]])
    #out = cv2.VideoWriter(fileout,fourcc, 20.0, (hsi.shape[0],hsi.shape[1]))
    out = cv2.VideoWriter(fileout+".avi",fourcc, 20.0, (xn.shape[0],yn.shape[0]))
    out2 = cv2.VideoWriter(fileout+"_normbyframe.avi",fourcc, 20.0, (xn.shape[0],yn.shape[0]))

    
    for k in range(0,hsi.shape[2],1):
        frame0 = hsi[:,:,k]
        #finterp = interp2d(x, y, frame0, kind='cubic')
        finterp = interp2d(y, x, frame0, kind='cubic')
        frameI  = finterp(yn,xn)
        frameI  = np.array([frameI,frameI,frameI]).T # 3 channels equal for gray in RGB
        
        frame   = np.uint8(abs(frameI)/(hsi.max()+.001)*255.0).copy()
        frame2   = np.uint8(abs(frameI)/(frameI.max()+.01)*255.0).copy()        
        
        # write the flipped frame
        wave = 345.500+k*0.21762930468776176
        
        cv2.putText(frame2,"%.1fnm" % (wave) ,(int(Nxout)-110,25), font, .6,(0,0,255),1,cv2.LINE_AA)
        cv2.putText(frame,"%.1fnm" % (wave) ,(int(Nxout)-110,25), font, .6,(0,0,255),1,cv2.LINE_AA)
        
        out.write(  frame )
        out2.write( frame2 )

        cv2.imshow('frame',frame2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    #cap.release()
    out.release()
    cv2.destroyAllWindows()


############


def factors(n):
    """ ehh this ignores double factors"""
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def freqs(n):
    """Fourier freq cooedinates in normal order (unlike np.fft.freq )"""
    return  np.sort(np.fft.fftfreq(n))


def split_vector(x,l,reshape=None):
    """ split vector into l-parts and put parts into a list 
    be sure that len(x) is ddivisible by l
    x - vec 
    l -  how many splits
    """
    n = len(x)
    m = n/l
    if reshape:
        return [x[(k*m):(k*m+m)].reshape(reshape) for k in range(l) ]        
    else:
        return [x[(k*m):(k*m+m)] for k in range(l) ]


def peak_signal_to_noise_ratio_dB(im_d,im_s,normalize=False):
    """ PSNR in dB for reconstruction quality evaluation -> see for ex. ZZ thesis
    im_d -> true data, im_s -reconstructed img
    """
    if normalize:
        im_d = im_d/ np.sqrt( (im_d**2).sum())
        im_s = im_s/ np.sqrt( (im_s**2).sum())        
    pnml = np.prod(im_d.shape)
    psnr = 10*np.log10( abs(im_d.max())**2/( abs(im_d.ravel()-im_s.ravel())**2/pnml).sum())
    #    psnr = 10*np.log10( im_d.max()**2/((im_d-im_s)**2).sum()/ len(im_d))
    return psnr

psnr = peak_signal_to_noise_ratio_dB 

def reconstruction_compresion_ratio(f,fi,x):
    """ How different representation compress the data
    f is function transfering to new basis
    and fi is function tranfering back
    """
    y0 =  f(x)
    y  = y0.ravel()
    idx = abs(y).argsort()
    n=len(idx)
    psnr=[]
    for k in np.linspace(1,n,100).astype(np.int64):
        # copy F representation
        yk = y.copy() + y.max()*np.random.randn(*y.shape)*1e-12 # add a bit of noise (in the case of many same val elements)
        # k smallest factors set to zero
        yk[idx[0:k]]*=0
        # and do the reconstruction
        xk = fi(yk)
        # calculate reconstruction quality:
        psnr.append(peak_signal_to_noise_ratio_dB(x,xk))
    return  psnr

compresion_ratio = reconstruction_compresion_ratio


def dotProductTest(Af,ATf,dimf,fil=False,seed=False):
    """ test if Af and ATf are adjont
    see: /EARTH SOUNDINGS ANALYSIS: Processing versus Inversion (PVI), 1992. Signal analysis. Introduction to inversion./  Jon Claerbout
    page 109 (pdf 128)
    http://sepwww.stanford.edu/sep/prof/
    """
    import time
    from scipy.ndimage import  gaussian_filter
    if seed:
        np.random.seed(999)
    #x = np.random.standard_normal(dimf)
    x = np.random.random_sample(dimf) # we do not like negative vals
    if fil: # smooth x with gaussian filter with width fil (some operators not handles fas varing signals)
        x=gaussian_filter(x,fil)
    t0_fw = time.time()
    Axx = Af(x)
    tfw = time.time() - t0_fw
    
    #    y = np.random.standard_normal(Axx.shape)
    y = np.random.random_sample(Axx.shape)
    if fil: # smooth x with gaussian filter with width fil (some operators not handles fas varing signals)
        y=gaussian_filter(y,fil)

    t0_adj = time.time()
    Byy  = ATf(y)    
    tadj = time.time() - t0_adj
        

    # make the comparison Axx . y with Byy . x
    dot1 = np.dot(Axx.ravel(),y.ravel())
    dot2 = np.dot(Byy.ravel(),x.ravel())

    print( "**********************************************************************\nDot Product Test:")
    #print "Ax.y ; A'y.x ", dot1,dot2, "ratio:",dot1/dot2, "rel diff: ",2*(dot1-dot2)/(dot1+dot2)
    print("Ax.y  :",  dot1)
    print( "A'y.x :",  dot2)
    print ("\nratio %.12f:" % (dot1/dot2))
    print("relative diff %g: " % (2*(dot1-dot2)/(dot1+dot2)))
    print("**********************************************************************")
    print("")
    print("exec time:")
    print("\t forward [sec]:\t", tfw)
    print("\t adjoint [sec]:\t", tadj)
    
    return Axx,Byy,x,y, (2*(dot1-dot2)/(dot1+dot2))

dp_test = dotProductTest


def dos2unix(f):
    """\
    convert dos linefeeds (crlf) to unix (lf) + decodes ISO encoding if needed
    usage: dos2unix(fileneme)
    returns StringIO readable by genfromtxt (no need encoding....) 
    
    """

    import sys
    from io import StringIO

    import subprocess
    out = subprocess.check_output(["file", f]).decode()
    i1=out.find(":")+2
    i2=out.find("text")-1
    encoding=out[i1:i2].strip().lower()
    if encoding[0] =='i':
        encoding+='-2'
    print("detected file: ", encoding)
    content = ''
    outsize = 0
    with open(f, 'rb') as infile:
        content = infile.read()
    output=b''
    #with open(sys.argv[2], 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output += ( line + b'\n')
        
    output_s = output.decode(encoding=encoding)     
    return StringIO(output_s)
    #print("Done. Stripped %s bytes." % (len(content)-outsize))
