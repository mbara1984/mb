import pywt
import numpy as np

class wavelets():
    """wavelets class - this version support all dimensions and samplings"""
    def __init__(self,nml,name='db1',mode='per',nscales=0):
        # ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization']
        self.wbasis = pywt.Wavelet(name)
        self.nml=nml
        self.shape=nml        
        self.mode=mode
        
        # if not is_power_of2(nml):
        #     print("CANNOT WORK WITH THIS SIZE OF ARRAY, use array of sizes 2^n")
        #     return -1 coJa@2018ZROBIE

        if nscales==0 :
            # how many level if sampling anisotropic?? minimal?? or scale diffin all cases
            self.nscales = pywt.dwt_max_level(data_len=min(self.nml), filter_len=self.wbasis.dec_len)
        else:
            self.nscales=nscales
        print("wavelet levels:",  self.nscales)
        
        z  = np.zeros(nml)+1.0
        self.cfs_template = pywt.wavedecn(z, self.wbasis, mode=self.mode, level=self.nscales)

            
    def decompose(self,x):
        """ w=B.x  """
#        print("X wavelet shape" , x.shape)
        x=x.reshape(self.nml)

        coeff_dict = pywt.wavedecn(x, self.wbasis, mode=self.mode, level=self.nscales)
        # Transform coef list to np.array (1D vector):
        cfs_vec   = wavelet_coefficients_list2vec(coeff_dict)

        # length_of_array = np.prod(self.size_list[0])
        # assert isinstance(coeff_arr, (np.ndarray))
        # length_of_array += sum(7 * np.prod(shape) for shape in size_list[1:-1])
        # assert len(coeff_arr) == length_of_array

        return cfs_vec

    def reconstruct(self,cfs_vec):
        """ x=B.w  """
        # Transform cfs_vec vector into list with dicts that is understood by pywt
        cfs1 = wavelet_coefficients_vec2list(cfs_vec.ravel(),self.cfs_template)
        # reconstruct: 
        xx =  pywt.waverecn(cfs1, self.wbasis, mode=self.mode)
        return xx

#######################################




def act_on_spatial_part(f,hsi,shape=None):
    if shape: # often data cube passed through adj operators is not prop shape
        hsi=hsi.reshape(shape)
    else:
        shape=hsi.shape
    hsi_g = np.zeros_like(hsi) #ASSUMING SQUARE (orthigonal transformation)

    for i in range(0,hsi.shape[2]):        
                hsi_g[:,:,i] = f(hsi[:,:,i].ravel()).reshape(shape[0],shape[0])
    return hsi_g

def act_on_spectral_part(f,hsi,shape=None):
    if shape: # often data cube passed through adj operators is not prop shape
        hsi=hsi.reshape(shape)
    else:
        shape=hsi.shape
    hsi_g = np.zeros_like(hsi) # ASSUMING SQUARE (orthigonal transformation)
    for i in range(0,hsi.shape[0]):
            for j in range(0,hsi.shape[1]):
                hsi_g[i,j,:] = f(hsi[i,j,:]) #.reshape(shape[0],shape[0])
    return hsi_g

# ex:
# basisSpace = wavelets.wavelets(hsi.shape[0:2],name='haar',mode='per',nscales=0)
# basisSpect = wavelets.wavelets(hsi.shape[2:],name='db3',mode='per',nscales=0)

# hsi2=act_on_spatial_part(basisSpace.decompose,hsi,hsi.shape)
#   act_on_spectral_part(basisSpect.decompose,hsi2,hsi.shape)

# act_on_spatial_part(basisSpace.reconstruct,hsi,hsi.shape)
# act_on_spectral_part(basisSpect.reconstruct,hsi2,hsi.shape)


# basisSpace = wavelets.wavelets(hsi.shape[0:2],name='haar',mode='per',nscales=0)
# basisSpect = wavelets.wavelets(hsi.shape[2:],name='db3',mode='per',nscales=0)
# Bf  = lambda XXX: act_on_spatial_part(basisSpace.decompose,act_on_spectral_part(basisSpect.decompose,XXX,hsi.shape),hsi.shape)
# BTf = lambda WWW: act_on_spatial_part(basisSpace.reconstruct,act_on_spectral_part(basisSpect.reconstruct,WWW,hsi.shape),hsi.shape)

class doubleWavelets():
    def __init__(self,nml,levels=0):        
        self.basisSpace = wavelets(nml[0:2],name='haar',mode='per',nscales=levels)
        self.basisSpect = wavelets(nml[2:],name='db3',mode='per',nscales=levels)
        self.nml = nml
    def decompose(self,XXX):
        return act_on_spatial_part(self.basisSpace.decompose,act_on_spectral_part(self.basisSpect.decompose,XXX,self.nml),self.nml)
    def reconstruct(self,WWW):
        return act_on_spatial_part(self.basisSpace.reconstruct,act_on_spectral_part(self.basisSpect.reconstruct,WWW,self.nml),self.nml)

class spaceWavelets():
    def __init__(self,nml,name='haar',mode='per',levels=0):        
        self.basisSpace = wavelets(nml[0:2],name=name,mode=mode,nscales=levels)
        self.nml = nml
    def decompose(self,XXX):
        return act_on_spatial_part(self.basisSpace.decompose,XXX,self.nml)
    def reconstruct(self,WWW):
        return act_on_spatial_part(self.basisSpace.reconstruct,WWW,self.nml)

class null():
    def __init__(self,nml,levels=0):        
        self.nml = nml
    def decompose(self,XXX):
        return XXX
    def reconstruct(self,WWW):
        return WWW



    


#######################################
class frame_as_union():
    """ class for union of wavelets/DCT's -> redundant frames """
    def __init__(self,shape, base_list,dtype=np.float64 ):
        """ shape is X shape, base_list is a LIST of transforms (all with decompose() and reconstruct() methods)  """
        self.nml   = shape
        self.shape = shape
        self.alpha = [] # wavelet coef holder (preallocated!)
        
        self.redundancy = len(base_list) # redundancy level (how many ortho bases we use)
        self.wave_list = base_list
        self.dtype=dtype
        for wav in range(len(base_list)):
            self.alpha.append(np.zeros(shape,dtype=dtype)) # allocate space for coefficients

        
    def synthesis(self,alpha): #  x = W.alpha (from coefs to x)
        
        x = np.zeros(self.shape,dtype=self.dtype) # our output
        if not (type(alpha) == list):
            #sl = [slice(0,None,1)]*(0+len(self.shape))            
            alpha =  alpha.reshape(self.redundancy, np.prod(self.shape))
            for k in range(self.redundancy):
                #sl[0]=k
                #x+= self.wave_list[k].reconstruct(alpha[sl])#/self.redundancy
                x+= self.wave_list[k].reconstruct(alpha[k,:]) #/self.redundancy                                
        else:
            for k in range(self.redundancy):
                print(alpha[k].shape)
                x+= self.wave_list[k].reconstruct(alpha[k])#/self.redundancy

        return x/self.redundancy*np.sqrt(self.redundancy)
    
    def analysis(self,x):        
        
        for k in range(self.redundancy):
            self.alpha[k] = self.wave_list[k].decompose(x).reshape(self.shape)
        out = np.array(self.alpha)
        out =out.ravel()/np.sqrt(self.redundancy)#*self.redundancy
        return out

    
class union_of_wavelets():
    """ class for unionof wavelets -> redundant frames """
    
    def __init__(self,shape,wlet_par=[['haar',0],['db3',0]]):
        """
        inputs:
        1. shape of data
        2. wlet_par = [[wavelet, levels],[wavelet, levels],...]
        levels = 0 -> finds maximal possible wavelet level number
        """
            #wlet_par=[['db1',0],['db1',1]]
        #[['db1',0],['db1',1]]
        
        self.nml = shape
        self.shape = shape
        self.alpha = [] # wavelet coef holder (preallocated!)
        
        self.redundancy = 0 
        self.wave_list=[]
        for wav in wlet_par:
            self.wave_list.append(wavelets(shape, wav[0], nscales=wav[1]))
            self.redundancy+=1
            self.alpha.append(np.zeros(shape)) # allocate space for coefficients
                
    def synthesis(self,alpha): #  x = W.alpha (from coefs to x)
        x = np.zeros(self.shape) # our output
        if not (type(alpha) == list):
            sl = [slice(0,None,1)]*(0+len(self.shape))            
            alpha =  alpha.reshape(self.redundancy,np.prod(self.shape)) # -> [R,n*m*l]

            for k in range(self.redundancy):
                sl[0]=k
                #x+= self.wave_list[k].reconstruct(alpha[sl])/self.redundancy
                x+= self.wave_list[k].reconstruct(alpha[k,:]) #/self.redundancy                
        else:
            for k in range(self.redundancy):
                print(alpha[k].shape)
                x+= self.wave_list[k].reconstruct(alpha[k]) #/self.redundancy
        return x/self.redundancy*np.sqrt(self.redundancy)
    
    def analysis(self,x):        
        for k in range(self.redundancy):
            self.alpha[k] = self.wave_list[k].decompose(x).reshape(self.shape)
        return np.array(self.alpha).ravel()/np.sqrt(self.redundancy)

    def decompose(self,alpha):
        return self.analysis(alpha)
    def reconstruct(self,x):
        return self.synthesis(x)
    
#######################################################################################
    def test_union_of_wavelets(self):

        #uw = self.union_of_wavelets([128],[0],['haar','db2'])
        # list: [[wavelet, levels],[wavelet, levels],...]
        import wavelets as wv
        reload(wv)
        uw = self.union_of_wavelets([128],[['haar',0],['db2',2]])
        uw = wv.union_of_wavelets([128],[['haar',0],['db2',0],['db8',0]])        
        uw.redundancy
        x = randn(128)
        x[x<0]=0

        alp = uw.analysis(x)    
        xx  = uw.synthesis(alp) 

        alp = uw.decompose(x)    
        xx  = uw.reconstruct(alp)

        
        plot(x)
        plot(xx)
        r2=1 #sqrt(2)
        Wf  = lambda x : uw.analysis(x) 
        WTf = lambda w : uw.synthesis(w)
        z   = utils.dotProductTest(Wf,WTf, (128)) # pass at: relative diff 3.19298e-16

        

        # test union of different frames

        # make set of WAVELETS
        uw = wv.union_of_wavelets([128],[['haar',0],['db2',0],['db8',0]])        
        # make Fourier transform Frame
        ft =  wv.fourier()
        # make DCT
        dctF =  wv.dct_m()
        
        frames_list = list(uw.wave_list)
#        frames_list.append(ft)
        frames_list.append(dctF)
        
        frames_list[-2].decompose(x)
        
        
        FRAMEs = wv.frame_as_union(x.shape,frames_list)

        y=FRAMEs.analysis(x)
        xx=FRAMEs.synthesis(y)
        FRAMEs.redundancy,        FRAMEs.shape
        len(FRAMEs.alpha),        len(FRAMEs.wave_list)
        #plot(x); plot(xx/2.,'o');
        #plot(alp[0],'o-');    plot(-alp[1],'-')

    
# helper function for PYWT coeff transform outside the class 


def wavelet_coefficients_list2vec(coeffs):
    """transform wavelets (from pywt) coefficient list structure to 1D np.array """
    num_cfs = count_wavelet_cfs(coeffs)

    a = np.zeros(num_cfs)
    # 0th order:
    nn = np.prod(coeffs[0].shape)
    a[0:nn] = coeffs[0].ravel()

    n1=nn; n2=nn
    for k in range(1, len(coeffs)): # loop over list
        ns = np.prod(coeffs[k][list(coeffs[1].keys())[0]].shape)
        for key in list(coeffs[k].keys()): # loop over dictionary fields
            a[n1:(n2+ns)] = coeffs[k][key].ravel()
            n1 +=ns; n2+=ns
    return a

def wavelet_coefficients_vec2list(cfs_vec,coeffs):
    """transform back wavelets  coefficient np.array to list (pywt) structure
       coeffs is correct list structure it will be OVERWRITTEN!
    """
    a= cfs_vec.copy()
    num_cfs = len(a)
    # 0th order:
    
    nn = np.prod(coeffs[0].shape)
    coeffs[0] = a[0:nn].reshape(coeffs[0].shape)
    #
    n1=nn; n2=nn
    for k in range(1, len(coeffs)):  # loop over first order list
        ashape = coeffs[k][list(coeffs[1].keys())[0]].shape
        ns = np.prod(ashape)          
        for key in list(coeffs[k].keys()): # loop over dict
            coeffs[k][key]=a[n1:(n2+ns)].reshape(ashape)
            n1 +=ns; n2+=ns
    return coeffs

def count_wavelet_cfs(coeffs): 
    """ counts number of wavelet coefs (stored in nested list/dicts)"""
    tt = np.prod(coeffs[0].shape) # coeffs[0] is allways np.array
    for k in range(1, len(coeffs)):  # we have list of dicts (level deps)
        for key in coeffs[k].keys(): # we have dict here
             tt += np.prod(coeffs[k][key].shape) # add total num of elems
    return tt

from scipy.fftpack import dct, idct        
class dct_m():
    """my DCT class - this version support all dimenstions and or samplings"""
    
    def __init__(self,shape,dct_type=2):
        self.dct_type=dct_type
        self.shape = shape
        self.nml = shape
    def decompose(self,x):
        y = x.reshape(self.nml).copy()
        for k in range(x.ndim):
            y = dct(y,axis=k,norm='ortho',type=self.dct_type)
        return y
    
    def reconstruct(self,y):
        x = y.reshape(self.nml).copy()
        for k in range(x.ndim):
            x = idct(x,axis=k,norm='ortho',type=self.dct_type)
        return x

    def testIt():
        import sys
        sys.path.append('/home/mbara/work/invs')
        import utils
        dw = dct_m(2)
        x = rand(128)
        x[12:33]=1
        y  = dw.decompose(x)

        dw2 = dct_m(3)
        y1  = dw2.decompose(x)
        plot(y1);plot(y)
        plot(y1-y)
        xx = dw.reconstruct(y)        
        plot(xx);plot(x,'o')

        # for i in range(120):
        #     x=dw.decompose(x)
        #     plot(x)
        
        z = utils.dotProductTest(dw.decompose,dw.reconstruct, (128)) # pass at 4.7e-16
        z = utils.dotProductTest(dw.decompose,dw.reconstruct, (128,128,64)) # pass at 6.7e-15                 

class fourier():
    """my FT class - this version support all dimenstions and or samplings"""
    
    def __init__(self,shape):
        self.shape = shape
        self.nml = shape        
        
    def decompose(self,x):
        return  np.fft.fftn(x.reshape(self.shape),norm='ortho')
        
    def reconstruct(self,y):
        return  np.fft.ifftn(y.reshape(self.shape),norm='ortho')

# class chirp():
#     import sys
#     sys.path.append('/home/mbara/soft/git/chirplet')
#     import chirplets
    
# from ltfatpy import greasy
# from ltfatpy import dgt,idgt # GABOR Transform

# class gabor(): # this is windowed FT https://en.wikipedia.org/wiki/Gabor_transform
#     def __init__(self,shape): # # is not a frame !
#         self.shape=shape
#         self.nml=shape
#         self.a = 32  # time shift
#         self.M = 64  # frequency shift
#         self.gs = {'name': 'blackman', 'M': 128}  # synthesis window
#         # analysis window
#         self.ga = {'name' : ('dual', self.gs['name']), 'M' : 128}
#         f= np.zeros(self.shape)
#         (c, Ls) = dgt(f, self.ga, self.a, self.M)[0:2]  # analysis
#         self.out_shape=c.shape
        
#     def decompose(self,f):
#         (c, Ls) = dgt(f, self.ga, self.a, self.M)[0:2]  # analysis
#         return c.ravel()
    
#     def reconstruct(self,c):
#         r = idgt(c.reshape(self.out_shape), self.gs, self.a)[0]  # synthesis
#         return r


# def gabor_fn(sigma, theta, Lambda, psi, gamma):
#     """ Gabor Filter from: https://en.wikipedia.org/wiki/Gabor_filter """
#     sigma_x = sigma
#     sigma_y = float(sigma) / gamma

#     # Bounding box
#     nstds = 3
#     xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
#     xmax = np.ceil(max(1, xmax))
#     ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
#     ymax = np.ceil(max(1, ymax))
#     xmin = -xmax
#     ymin = -ymax
#     (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

#     # Rotation 
#     x_theta = x * np.cos(theta) + y * np.sin(theta)
#     y_theta = -x * np.sin(theta) + y * np.cos(theta)

#     gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
#     return gb


# def cwt():
#     pass
#     # >>> from scipy import signal
#     # >>> import matplotlib.pyplot as plt
#     # >>> t = np.linspace(-1, 1, 200, endpoint=False)
#     # >>> sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
#     # >>> widths = np.arange(1, 31)
#     # >>> cwtmatr = signal.cwt(sig, signal.ricker, widths)
#     # >>> plt.imshow(cwtmatr, extent=[-1, 1, 31, 1], cmap='PRGn', aspect='auto',
#     #                    vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())


if __name__ == '__main__':

    import spectral_object
    from   pylab import imread
    import numpy as np

    import pprint
    import wavelets3d
    reload(wavelets3d)
    argb = imread("FIGS/lena_std.png")[::8,::8,:]
    mrgb = spectral_object.rgb2multispectral(argb,64)
    n=32 # IT DOES HAVE to be a multiple of 2 !!!!!!!!!!!!!!!!!!!
    mrgb = mrgb[0:n,0:n,0:n]
    mrgb.shape
    wavlet= wavelets3d.wavelet3d(mrgb.shape,name='haar',mode='per',nscales=0)
    nml=wavlet.nml
    wavlet.size_list
    w = wavlet.decompose(mrgb);     print(w.shape, "/tot", np.prod(nml))

    x = wavlet.reconstruct(w)


    del wavlet
    #    pprint.pprint(result)
