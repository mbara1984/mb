#from lfis import utils
import numpy  as np
from scipy.signal import convolve2d # (in1, in2, mode='full', boundary='fill', fillvalue=0)

class random():
    def __init__(self,shape,m,seed=666):
        self.nm = np.prod(shape)
        self.shape = shape
        self.m= m
        self.seed=seed

    def forward(self,img):

        np.random.seed(self.seed)
        out = np.zeros(self.m)
        for m in range(self.m):
            v = np.random.rand(self.nm)
            v = v/np.sqrt((v**2).sum())
            out[m] = v.dot(img.ravel())

        return out
    
    def adjoint(self,ou, reshape=None):

        np.random.seed(self.seed)

        img = np.zeros(self.nm)

        for m in range(self.m):
            v = np.random.rand(self.nm)
            v = v/np.sqrt((v**2).sum())            
            img += v* ou[m]

        if reshape is None:
            return img
        else:
            return img.reshape(self.shape)
        
class randomBinary():
    def __init__(self,shape,m,seed=666):
        self.nm = np.prod(shape)
        self.shape = shape
        self.m= m
        self.seed=seed

    def forward(self,img):

        np.random.seed(self.seed)
        out = np.zeros(self.m)
        img = img.ravel()
        for m in range(self.m):
            #v = np.random.randint(0,2,self.nm)
            v= np.random.permutation([-1,1]*self.nm)
            #v = v/np.sqrt((v**2).sum())
            #out[m] = v.dot(img.ravel())
            out[m] = np.einsum("i->",img[v])
        return out
    
    def adjoint(self,ou, reshape=None):

        np.random.seed(self.seed)

        img = np.zeros(self.nm)

        for m in range(self.m):
            #v = np.random.rand(self.nm)
            #v = np.random.randint(0,2,self.nm)
            v= np.random.permutation([-1,1]*self.nm)            
            #v = v/np.sqrt((v**2).sum())            
            img[v] += ou[m]

        if reshape is None:
            return img
        else:
            return img.reshape(self.shape)
        
class randomBlock():
    def __init__(self,shape,m,seed=666,step=16):
        self.nm = np.prod(shape)
        self.shape = shape
        self.m = m
        self.seed=seed
        self.step = step
    def forward(self,img):

        np.random.seed(self.seed)
        out = np.zeros(self.m)
        k=0
        for m in range(self.m):
            v = np.random.rand(self.nm/self.step)            
            v = v/np.sqrt((v**2).sum())
            out[m] = v.dot(img.ravel()[k::self.step])
            k+=1
            if k>=self.step:
                k=0
        return out
    
    def adjoint(self,ou, reshape=None):

        np.random.seed(self.seed)

        img = np.zeros(self.nm)
        k=0
        for m in range(self.m):
            v = np.random.rand(self.nm/self.step)
            v = v/np.sqrt((v**2).sum())            
            img[k::self.step] += v*ou[m]
            k+=1
            if k>=self.step:
                k=0

        if reshape is None:
            return img
        else:
            return img.reshape(self.shape)

        
class randomSin():
    """twise slower as rand """
    def __init__(self,shape,m,seed=666):
        self.nm = np.prod(shape)
        self.shape = shape
        self.m= m
        self.seed=seed
        self.x= np.linspace(0,1,self.nm)

    def forward(self,img):

        np.random.seed(self.seed)
        out = np.zeros(self.m)
        for m in range(self.m):
            phase = np.random.rand()*np.pi*2
            freq  = np.random.rand()*self.nm/2.0
            v = np.sin(self.x*freq+phase)
            #v = v/np.sqrt((v**2).sum())            
            out[m] = v.dot(img.ravel())

        return out
    
    def adjoint(self,ou, reshape=None):
        np.random.seed(self.seed)
        img = np.zeros(self.nm)

        for m in range(self.m):
            phase = np.random.rand()*np.pi*2
            freq  = np.random.rand()*self.nm/2.0 
            v     = np.sin(self.x*freq+phase)
            #v = v/np.sqrt((v**2).sum())
            img += v* ou[m]

        if reshape is None:
            return img
        else:
            return img.reshape(self.shape)

#from lfis import utils
#w=utils.dotProductTest(fw,adj,img.shape)


class conv2d():
    def __init__(self,psf):
        self.psf = psf
        self.n = psf.shape[0]
        self.m = psf.shape[1]
        
    def forward(self,img):
        return convolve2d(img,self.psf)
    
    def adjoint(self,img):
        return convolve2d(img,self.psf[::-1,::-1])[(self.n-1):-(self.n-1),(self.m-1):-(self.m-1)] #[9:-9,9:-9]


class binning():
    def __init__(self,shape,order):
        self.order = order
        self.shape = shape
        self.new_shape = [ int(np.ceil(shape[0]/(order+0.0))*order), int(np.ceil(shape[1]/(order+0.0))*order) ]
        self.img = np.zeros(self.new_shape)
        
    def forward(self,img):
        self.img[:self.shape[0],:self.shape[1]]=img
        
        n = self.order
        a = 0*self.img[::n,::n].copy()
        for k in range(0,n):
            for kk in range(0,n):
                a+=  self.img[kk::n,k::n]
        return a.copy()

    def adjoint(self,img):
        n = self.order
        
        img2 = np.zeros((img.shape[0]*n,img.shape[1]*n))
        for k in range(0,n):
            for kk in range(0,n):
                img2[kk::n,k::n] = img
        return img2[:self.shape[0],:self.shape[1]].copy()


class pixel_shift():
    def __init__(self,shape,xy):
        """ xy respect to center """
        self.shape  = shape
        xmi = xy[:,0].min(); xmx = xy[:,0].max()
        ymi = xy[:,1].min(); ymx = xy[:,1].max()        
        
        self.new_shape = [shape[0]  + (ymx - ymi),shape[1]  + (xmx - xmi) ]
        self.x = xy[:,0]-xmi
        self.y = xy[:,1]-ymi 
        self.xy  = xy
        self.out = np.zeros(self.new_shape)
        self.adj = np.zeros(self.shape)
        
    def forward(self,img):
        self.out*=0
        for k in range(len(self.x)):
            self.out[self.y[k]:(self.y[k]+self.shape[0]),self.x[k]:((self.x[k]+self.shape[1]))] += img
        return self.out.copy()

    
    def adjoint(self,img):
        self.adj*=0
        for k in range(len(self.x)):
            self.adj += img[self.y[k]:(self.y[k]+self.shape[0]),self.x[k]:((self.x[k]+self.shape[1]))]
        return self.adj.copy()

def main_shift():
    fname= "/home/mbara/soft/matlab/osdl/lena.png"
    img=imread(fname)[::4,::4]
    
    #ss=pixel_shift(img.shape,np.array([[40,30,10,44],[1,12,51,44]]).T )

    x,y = np.mgrid[0:100:15,0:100:15]

    ss = pixel_shift(img.shape,array([x.ravel(),y.ravel()]).T )
    
    y  = ss.forward(img)
    
    Np = 10000*y.shape[0]*y.shape[0]
    yP = np.random.poisson(y/y.sum()*Np )
    xP = ss.adjoint(yP)
    xx = ss.adjoint(y)    
    
    import pySPIRALTAP as sprl
    res= sprl.SPIRALTAP(yP,ss.forward,1e-4,AT=ss.adjoint,maxiter=1000,miniter =50)



    from apg import apg as apg_
    reload(apg_)
    xap = apg_.apg_simple(ss.forward, ss.adjoint, yP+0.0, img.shape, 1e-8, max_iters=1000)
    matshow(yP+0.0)
    matshow(xap[0].reshape(img.shape))

def main_psf():
    fname= "/home/mbara/soft/matlab/osdl/lena.png"
    img=imread(fname)[::4,::4]

    psf = np.random.rand(6,6)
    psf=psf/sqrt(psf.sum())
    ss =  conv2d(psf)

    y  = ss.forward(img)
    
    Np = 1000*y.shape[0]*y.shape[0]
    yP = np.random.poisson(y/y.sum()*Np )
    xP = ss.adjoint(yP)
    xx = ss.adjoint(y)    


    from lfis import wavelets
    basis = wavelets.wavelets(img.shape, name='haar', mode='per')
    w  = basis.decompose(img)   # B
    xx = basis.reconstruct(w) # BT
    
    import pySPIRALTAP as sprl
    #res= sprl.SPIRALTAP(yP,ss.forward,1e-4,AT=ss.adjoint,maxiter=1000,miniter =50)

    #Af  =  lambda w: ss.forward(basis.reconstruct(w))
    #ATf =  lambda x: basis.decompose( ss.adjoint(x) )
    #res= sprl.SPIRALTAP(yP,ss.forward,1,AT=ss.adjoint,maxiter=100,W=basis.decompose,WT=basis.reconstruct,miniter =50)
    WT = lambda x: basis.decompose(x).reshape(img.shape)
    res= sprl.SPIRALTAP(yP,ss.forward,.02,AT=ss.adjoint,maxiter=100,WT=WT,W=basis.reconstruct,penalty='onb', miniter =50,savereconerror=True,saveobjective=True,truth=img,verbose=2)
    matshow(res[0])

    
    from apg import apg as apg_
    reload(apg_)
    #xap = apg_.apg_simple(ss.forward, ss.adjoint, yP+0.0, img.shape, 1e-8, max_iters=1000)
    mu=30
    apg_opts = apg_.apg_options(max_iters=1000, apg_eps=1e-9)
    #apg_opts.min_iters=500
    xap,err = apg_.apg_solve_AB(ss.forward, ss.adjoint, basis.decompose, basis.reconstruct, yP+0.0,img.shape, mu, apg_opts = apg_opts )

    #xap,err = apg_.apg_simple(Af, ATf, yP+0.0, img.shape, 1e-8, max_iters=1000)

    
    matshow(yP+0.0)
    matshow(xap.reshape(img.shape))



def main_denoise():
    fname= "/home/mbara/soft/matlab/osdl/lena.png"
    img=imread(fname)[::4,::4]
    

    fwd = lambda x: 2*x+0.00
    adj = lambda x: 2*x+0.00
    
    y  = fwd(img)
    
    Np = 10*y.shape[0]*y.shape[0]
    y= y/y.sum()*Np
    yP = np.random.poisson(y )
    imshow(yP)
    from lfis import wavelets
    basis = wavelets.wavelets(img.shape, name='haar', mode='per')
    basis = wavelets.wavelets(img.shape, name='db5', mode='per')
    
    import pySPIRALTAP as sprl
    WT = lambda x: basis.decompose(x).reshape(img.shape)
    res= sprl.SPIRALTAP(yP,fwd,.075,AT=adj,maxiter=900,WT=WT,W=basis.reconstruct,penalty='tv', miniter =50,savereconerror=True,saveobjective=True,truth=y,verbose=2)
    matshow(res[0])
    matshow(res[0]-y)
    matshow(y)

    # APG
    mu=9
    yP= y+rand(*y.shape)*10.0
    apg_opts = apg_.apg_options(max_iters=1000, apg_eps=1e-9)
    apg_opts.min_iters=500
    xap,err = apg_.apg_solve_AB(fwd, adj, basis.decompose, basis.reconstruct, yP+0.0,img.shape, mu, apg_opts = apg_opts )
    matshow(xap.reshape(img.shape))
    matshow(yP)

