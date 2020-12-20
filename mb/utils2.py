## PLOTS AND GENERAL STUFF
    
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


def conjgrad(Af,b,x0,itmax=100,itmin=50,force_nonegative=False):
    """ simplest possible implementation of conjugate gradient method to solve Ax=b
    equation - in here matrix form is not needed since A.dot(x_probe) is used for eval
    - this algorithm is VALID ONLY for symmetric positive-definite systems 
    for different alg for more gen case see Kaczmarz Method

    Im using it for Tikonov inversion:
    
    mu = 0.02
    forward_Tikh = lambda x: (adjoint(forward(x))) + mu*x 
    b_Tikh = adjoint(a)
    w0,res = conjgrad(forward_Tikh,b_Tikh,s*0,itmax=300)
    """
    #print(x0.shape,b.shape)
    r = b-Af(x0);
    p = r;
    rsold=r.T.dot(r);
    x=x0
    rsnew = r.T.dot(r);
    res=[np.sqrt((r**2).sum())]
    for i in range(itmax):
        print( i, np.sqrt(rsnew).sum(),p.shape)
        Ap    = Af(p)          
        alpha = rsold/(p.dot(Ap));
        x = x+alpha*p;
        
        if force_nonegative:
            x[x<0]=0
        
        r = r-alpha*Ap;
        rsnew = r.T.dot(r);
        if i> itmin and  np.sqrt(rsnew).sum()<1e-10:
            break;
        p=r+(rsnew/rsold)*p;
        rsold=rsnew;
        #x=abs(x)
        #print(x.shape)
        res.append(np.sqrt((r**2).sum()))
    return x,res

def solveTikhonov(forward,adjoint,b,mu,x0,itmax=100, itmin=50, force_nonegative=False):
    # b - measurement
    #forward_Tikh = lambda x: (adjoint(forward(x))).ravel() + mu*x.ravel()
    forward_Tikh = lambda x: (adjoint(forward(x.reshape(x0.shape)))).ravel() + mu*x.ravel()
    
    b_Tikh = adjoint(b).ravel()
    w0,res = conjgrad(forward_Tikh,b_Tikh,x0.ravel(),itmax=itmax, itmin=itmin, force_nonegative=force_nonegative)    
    return w0.reshape(x0.shape), res


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

    print("**********************************************************************\nDot Product Test:")
    #print "Ax.y ; A'y.x ", dot1,dot2, "ratio:",dot1/dot2, "rel diff: ",2*(dot1-dot2)/(dot1+dot2)
    print("Ax.y  :",  dot1)
    print("A'y.x :",  dot2)
    print("\nratio %.12f:" % (dot1/dot2))
    print("relative diff %g: " % (2*(dot1-dot2)/(dot1+dot2)))
    print("**********************************************************************")
    print("")
    print("exec time:")
    print("\t forward [sec]:\t", tfw)
    print("\t adjoint [sec]:\t", tadj)
    
    return Axx,Byy,x,y, (2*(dot1-dot2)/(dot1+dot2))

def power_iteration(ATAf,shape,numiter=100):
    """ calculate dominant eigenvalue by POWER ITERATION method
        returns dominant eigenvalue:
        this is not allways convergent
        see https://en.wikipedia.org/wiki/Power_iteration
    """
    bi = np.random.randn(*shape)
    print(bi.shape)
    for i in range(numiter):
        print((i,np.linalg.norm(bi)))
        bi = ATAf(bi)
        bi = bi/np.linalg.norm(bi)
    bi = ATAf(bi)

    # check the sign: (by definition of eigenvalue Ax=lam*x)
    lam = np.linalg.norm(bi)
    v = ATAf(bi)/lam
    #print (bi/v).mean(), (bi/v).std()
    
    return lam*np.sign((bi/v).mean())
