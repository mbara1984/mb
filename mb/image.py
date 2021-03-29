import numpy  as np
import mb
pi =np.pi


###################################################
####### patch-wise operations
def fill_infinites(g):
    from scipy.interpolate import griddata
    x,y = mgrid[0:g.shape[0],0:g.shape[1]]
    we=griddata(np.array([x[np.isfinite(g)],y[np.isfinite(g)]]).T, g[np.isfinite(g)], np.array([x[~np.isfinite(g)],y[~np.isfinite(g)]]).T,method='nearest' )
    gg = g.copy()
    gg[~np.isfinite(g)]=we    
    matshow(gg)
    return gg



def local_normalization(img,p=256):
    img =img+0.0
    msk = img*0
    nrm = img*0
    for i in range(0,img.shape[0],p):
        for j in range(0,img.shape[1],p):
            msk[i:(p+i), j:j+p] = img[i:(p+i), j:j+p]/img[i:(p+i), j:j+p].max()
            nrm[i:(p+i), j:j+p] = img[i:(p+i), j:j+p].max()            
            #msk[i:(p+i), j:j+p] = img[i:(p+i), j:j+p]/median(img[i:(p+i), j:j+p])
            #msk[i*p:(1+i)*p,j*p:(j+1)*p] = img[i*p:(1+i)*p,j*p:(j+1)*p] < img[i*p:(1+i)*p,j*p:(j+1)*p].max()*maxratio
    return msk,nrm


def local_normalizationGlobal(img,p=256):
    img = img+0.0
    msk = img*0
    nrm = img*0
    for i in range(0,img.shape[0],1):
        for j in range(0,img.shape[1],1):
            msk[i:(p+i), j:j+p] += img[i:(p+i), j:j+p]/img[i:(p+i), j:j+p].max()
            nrm[i:(p+i), j:j+p] += img[i:(p+i), j:j+p].max()            
            #msk[i:(p+i), j:j+p] = img[i:(p+i), j:j+p]/median(img[i:(p+i), j:j+p])
            #msk[i*p:(1+i)*p,j*p:(j+1)*p] = img[i*p:(1+i)*p,j*p:(j+1)*p] < img[i*p:(1+i)*p,j*p:(j+1)*p].max()*maxratio
    return msk,nrm


def local_minmax(img,p=256):
    img =img+0.0
    msk = img*0
    nrm = img*0
    for i in range(0,img.shape[0],p):
        for j in range(0,img.shape[1],p):
            msk[i:(p+i), j:j+p] = (img[i:(p+i), j:j+p]-img[i:(p+i), j:j+p].min()) / (1e-9+img[i:(p+i), j:j+p] - img[i:(p+i), j:j+p].min()).max()
            nrm[i:(p+i), j:j+p] = img[i:(p+i), j:j+p].max()            
            #msk[i:(p+i), j:j+p] = img[i:(p+i), j:j+p]/median(img[i:(p+i), j:j+p])
            #msk[i*p:(1+i)*p,j*p:(j+1)*p] = img[i*p:(1+i)*p,j*p:(j+1)*p] < img[i*p:(1+i)*p,j*p:(j+1)*p].max()*maxratio
    return msk,nrm


def local_f(img,f,p=32):
    img =img+0.0
    msk = img*0
    nrm = img*0
    for i in range(0,img.shape[0],p):
        for j in range(0,img.shape[1],p):
            msk[i:(p+i), j:j+p] = f(img[i:(p+i), j:j+p])
    return msk


# ####### patch-wise operations
#####################################################################
