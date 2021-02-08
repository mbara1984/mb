import sympy
import numpy as np

"new stuff"
def abcdef(a,b,c,d,e,f):
    """ create 3x3 matrix (sympy object) for ABCDEF symbolic ray tracing"""
    return sympy.Matrix( ((a,b,e), (c,d,f), (0,0,1)) )

def chirp(f):
    a=1;b=0;c=-1/f;d=1;e=0; f=0
    return sympy.Matrix( ((a,b,e), (c,d,f), (0,0,1)) )

lens = chirp 

def prop(z):
    a=1;b=z;c=0;d=1;e=0; f=0
    return sympy.Matrix( ((a,b,e), (c,d,f), (0,0,1)) )

def shift(x0):
    a=1;b=0;c=0;d=1;e=x0; f=0
    return sympy.Matrix( ((a,b,e), (c,d,f), (0,0,1)) )

def tilt(alpha):
    a=1;b=0;c=0;d=1;e=0; f=alpha
    return sympy.Matrix( ((a,b,e), (c,d,f), (0,0,1)) )

def simpleABCDEF(M):
    """Algebraic simplification of Matrix analytical expresions"""
    A=sympy.simplify(M[0,0]); B=sympy.simplify(M[0,1])
    C=sympy.simplify(M[1,0]); D=sympy.simplify(M[1,1])
    E=sympy.simplify(M[0,2]); F=sympy.simplify(M[1,2])
    Ms=abcdef(A,B,C,D,E,F)
    return Ms

def propagation_abcdefg(M):
    # expecting x, u = sympy.symbols('x u')
    x, u = sympy.symbols('x u') 
    ray=sympy.Matrix( (x, u, 1) )
    r_out = M*ray
    return (r_out[0],r_out[1])

def substitute_params(sym_exp,params):
    """substitute use list of params=[[f0,f1],[f1,1.0]]"""
    f=sym_exp
    for par in params:
        f=f.subs(par[0],par[1])
    return f

### numeralisation

def propagate_num(M,xx,uu,alpha,x0_num=0):
    """ do the propagation through system M (abcdef matrix) 
    input rays defined by x,u (position, angle) (numpy.arrays) """

    xu = propagation_abcdefg(M)
#    print "propagation equation:"
    # lambdify use ORGINAL variables names in evaluation!
    x, u, alpha_g, x0 = sympy.symbols('x u alpha_g x0')
    xout_num = sympy.lambdify((x,u,alpha_g,x0),xu,'numpy')
        
    return xout_num(xx,uu,alpha,x0_num)


def ray_fan(M1,M_MLA,params,xxx,uuu,alpha,a):
    import numpy as np
    """ (M1,M_MLA,params,x,u,alpha_g,a)
    (x,u) - init ray coord, 
    a - lenslet size (assuming 100%FF) """
    
    # substitute system parameters (like f0,f1...)

    M1_num=substitute_params(M1,params)
    Mmla_num=substitute_params(M_MLA,params)
    
    # do numerical propagation of rays (x,u) ap to microlens array
    xmla,umla = propagate_num(M1_num,xxx,uuu,alpha)
    #print xmla
    # calculate the x0 (which microlens ray hits)
    # assuming odd number of lenslets (x0_i=i*a)   
    lenslet_n = np.round(xmla/a) # lenslet number
    print(lenslet_n)
    x0_num    = a*lenslet_n         # center coordinate of the (n-th) lenslet
    
    xout,vout = propagate_num(Mmla_num,xmla,umla,alpha,x0_num)

    return xout,vout

def abcdefplot(x,y,marker='o'):
    import matplotlib
    matplotlib.pylab.plot(x.flat,y.flat,marker,fillstyle='none',markersize=11,markeredgewidth=2)
    xlabel("$x_{in}/a$")
    ylabel("$x_{out}/a$")

def abcdefplotCol(x,y,marker='o',f=0.0):
    import matplotlib
    matplotlib.pylab.plot(x.flat,y.flat,marker,fillstyle='none',markersize=11,markeredgewidth=2,color=(f,1-f,1-f/2))
    xlabel("$x_{in}/a$")
    ylabel("$x_{out}/a$")

def abcdefplotColSize(x,y,marker='o',f=0.0,msize=11.0):
    import matplotlib
    matplotlib.pylab.plot(x.flat,y.flat,marker,fillstyle='none',markersize=msize,markeredgewidth=.75,color=(f,1.0-f,1.0-f/2.0))
    xlabel("$x_{in}/a$")
    ylabel("$x_{out}/a$")

    
    
def mxprint(mx):
    sympy.pretty_print(mx)
    print("----------------------------------------")
    print(sympy.latex(mx))
    print("----------------------------------------")
    return(sympy.latex(mx))

def process_M(Ml,name):
    mx = simpleABCDEF(Ml)
    print("M="+name)
    mxl=mxprint(mx)
    return mx
