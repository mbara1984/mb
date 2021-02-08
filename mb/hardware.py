import numpy as np
from pylab import *


def getSIFdata(filename):
    f    = open(filename, 'rb')
    allf = f.read()
    #n1   =  allf.find(b'Pixel')
    n1   =  allf.find(b'         ')
    ne   =  allf.find(b'<?xml')
    s=allf[n1+len(b'         '):ne]
    ss=frombuffer(s,dtype=np.float32)
    #plot(ss)
    return ss



def read_nanodrop(fname):
    """
    reads nanodrop tsv spectrum files 
    """
    ff = open(  fname) #fdir +'msc_mediaT.tsv'
    aa=[]
    s=''
    s_prev=''
    names=[]
    while True:
        s_prev_prev=s_prev
        s_prev=s
        s=ff.readline()    #< look for begining of data
        if s =='':
            break
        if s[0]=='W':
            names.append(s_prev_prev)
            arr=[]
            while True:
                r = ff.readline()
                if r =='':
                    break
                elif ord(r[0])<48 or ord(r[0])>92: 
                    break
                else:
                    w=np.float64(r.split('\t'))
                    arr.append(np.array([w[0],w[1]]))
            aa.append(array(arr))
    ff.close()
    return aa, names



def get_cary_flourescence_3Ddata(fname):
    """
    fdir = '/home/mbara/DATA/Uv_VIS_NIR_fluorescence/2020_12_PS_DMEM/maciej/'
    fname= dir +'dmem_nofbs.csv'

    """
    
    f=open(fname)
    h1=f.readline()
    h2=f.readline()
    data = []

    def mline(a):
        dta=[]
        while True:
            s=a[:a.find(',')]
            #print(s)
            dta.append(np.float64(s))
            a=a[a.find(',')+1:]
            if len(a)<2:
                break
        dta=np.array(dta)
        return dta[1::2],dta[::2]

    data = []
    ex   = []
    while 1:
        a=f.readline()
        if len(a)>1:
            try :
                emm, exx =mline(a)   
                data.append(emm)
                ex.append(exx[0])
            except ValueError:
                break

    while 1:
        a=f.readline()
        if a =='':
            break
        if a[:19]=='Ex. Wavelength (nm)':
            print(a)
            ex0=np.float(a[-8:])
        elif a[:14]=='Ex.  Stop (nm)':
            ex1=np.float(a[-8:])
            print(a)
        elif a[:19]=='Ex.  Increment (nm)':
            dex=np.float(a[-8:])
            print(a)

    exk=np.linspace(ex0-dex,ex1,int((ex1-ex0)/dex) )
    data=np.array(data)
    plot(ex,data)

    #matshow(data.T,extent=(ex[0],ex[-1],ex1,ex0));axis('auto')
    matshow(log(data.T+1),extent=(ex[0],ex[-1],ex1,ex0));axis('auto')
    ylabel('excitation [nm]')
    xlabel('emission [nm]')
    tight_layout()
    title('fluorescence: ' + os.path.basename(f))
    return data,(exk,ex1,ex0)
