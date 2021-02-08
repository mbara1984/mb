import sympy
import numpy as np

def stokes_vec(sym):
    s0,s1,s2,s3  = sympy.symbols(sym)   # retardance of lcc
    return sympy.Matrix(((s0 ),
                         (s1),
                         (s2),
                         (s3)))
def unpolarized():
    return sympy.Matrix(((1 ),
                         (0),
                         (0),
                         (0)))


def circular_right():
    return sympy.Matrix(((1 ),
                         (0),
                         (0),
                         (1)))

def circular_left():
    return sympy.Matrix(((1 ),
                         (0),
                         (0),
                         (-1)))

def linear_x():
    return sympy.Matrix(((1 ),
                         (1),
                         (0),
                         (0)))
def linear_y():
    return sympy.Matrix(((1 ),
                         (-1),
                         (0),
                         (0)))

def linear_45():
    return sympy.Matrix(((1 ),
                         (0),
                         (1),
                         (0)))

def linear_135():
    return sympy.Matrix(((1 ),
                         (0),
                         (-1),
                         (0)))


def mullerCONST(m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33):
    """ create 4x4 Muller matrix (sympy object) """
    #return sympy.Matrix( ((a,b,e,), (c,d,f), (0,0,1)) )
    # not 16parameters but? TODO
    return sympy.Matrix(((m00,m01,m02,m03),
                         (m10,m11,m12,m13),
                         (m20,m21,m22,m23),
                         (m30,m31,m32,m33)) )



def muller(m00,m01,m02,m03,m10,m11,m12,m13,m20,m21,m22,m23,m30,m31,m32,m33):
    """ create 4x4 Muller matrix (sympy object)
    with all constraints > 10
    #https://sci-hub.se/10.1007/978-3-540-74276-0_3
    """
    #return sympy.Matrix( ((a,b,e,), (c,d,f), (0,0,1)) )
    return sympy.Matrix(((m00,m01,m02,m03),
                         (m10,m11,m12,m13),
                         (m20,m21,m22,m23),
                         (m30,m31,m32,m33)) )


def make_rotation_matrix(theta):

    m22 = m11 = sympy.cos(2*theta)
    m21 =  -sympy.sin(2*theta)
    m12 =   sympy.sin(2*theta)        
    return sympy.Matrix(((1, 0 , 0 , 0 ),
                         (0,m11, m12, 0),
                         (0,m21, m22, 0),
                         (0, 0,  0 , 1)))
   
def polarizer(): # 0deg  axis
    return sympy.Matrix(((1, 1, 0, 0),
                         (1, 1, 0, 0),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0)))/2


def polarizer_circ_right(): # 0deg  axis
    return sympy.Matrix(((1, 0, 0, 1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0),
                         (1, 0, 0, 1)))/2
def polarizer_circ_left(): # 0deg  axis
    return sympy.Matrix(((1, 0, 0, -1),
                         (0, 0, 0, 0),
                         (0, 0, 0, 0),
                         (-1, 0, 0, 1)))/2


def retarder(delta): # linear 0deg  fast axis
    m22 =  m33 = sympy.cos(delta)
    m23 =  sympy.sin(delta)
    m32 = -sympy.sin(delta)                        
    return sympy.Matrix(((1, 0, 0, 0),
                         (0, 1, 0, 0),
                         (0, 0, m22, m23),
                         (0, 0, m32, m33)))


################################################# NO trig, complex exp

def retarder_notrig(delta): # linear 0deg  fast axis
    cos_delta = (sympy.exp(sympy.I*delta) + sympy.exp(-sympy.I*delta) )/2
    sin_delta = (sympy.exp(sympy.I*delta) - sympy.exp(-sympy.I*delta) )/(2*sympy.I)

    m22 =  m33 = cos_delta 
    m23 =  sin_delta
    m32 = -sin_delta                        
    return sympy.Matrix(((1, 0, 0, 0),
                         (0, 1, 0, 0),
                         (0, 0, m22, m23),
                         (0, 0, m32, m33)))

def make_rotation_matrix_notrig(delta):
    cos_delta = (sympy.exp(sympy.I*2*delta) + sympy.exp(-sympy.I*2*delta) )/2
    sin_delta = (sympy.exp(sympy.I*2*delta) - sympy.exp(-sympy.I*2*delta) )/(2*sympy.I)

    m22 = m11 = cos_delta
    m21 =  -sin_delta
    m12 =   sin_delta
    
    return sympy.Matrix(((1, 0 , 0 , 0 ),
                         (0,m11, m12, 0),
                         (0,m21, m22, 0),
                         (0, 0,  0 , 1)))

def rotate_operator_notrig(theta,Op):
    Ma =  make_rotation_matrix_notrig(theta)
    Mb =  make_rotation_matrix_notrig(-theta)   
    return Mb*Op*Ma


#################################################

def retarder_circular(delta): # 
    m11 =  m22 = sympy.cos(delta)
    m12 =  sympy.sin(delta)
    m21 = -sympy.sin(delta)                        
    return sympy.Matrix(((1, 0, 0, 0),
                         (0, m11, m12, 0),
                         (0, m21, m22, 0),
                         (0, 0, 0, 1)))


def rotate_operator(theta,Op):
    Ma =  make_rotation_matrix(theta)
    Mb =  make_rotation_matrix(-theta)   
    return Mb*Op*Ma

def efficiency_operator(eta,op):
    """ adds efficiency factor to muller operator: eta*Op + (1-eta)*I
    mainly to model nonperfect polarizers
    """
    return op*eta+(1-eta)*sympy.Matrix.diag(1,1,1,1)
    

def lucid(s_in):# lucid camera operator (4 polarizers)
    a0=rotate_operator(0,polarizer())           * s_in
    a90=rotate_operator(sympy.pi/2,polarizer())    * s_in
    a45=rotate_operator(sympy.pi/4,polarizer())    * s_in
    a135=rotate_operator(-sympy.pi/4,polarizer())  * s_in

    return sympy.Matrix([a0[0],a90[0],a45[0],a135[0]]) # returns measurement vector not stokes
    
def lucid_show(mbg):
    a=np.vstack([np.hstack([mbg[0],mbg[1]]),np.hstack([mbg[2],mbg[3]])])
    from pylab import matshow,axis
    matshow(a)

def substitute_params(sym_exp,params):
    """substitute use list of params=[[f0,f1],[f1,1.0]]"""
    f=sym_exp
    for par in params:
        f=f.subs(par[0],par[1])
    return f

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


def examples():

    d_sample = sympy.symbols('delta_sample')  # retardance of sample
    dlcc  = sympy.symbols('delta_lcc')   # retardance of lcc
    
    alpha = sympy.symbols('alpha') # rotation of analyzer
    th = sympy.symbols('theta_a') # rotation of analyzer

    rs  = stokes.retarder(d_sample)
    rs_rot = stokes.rotate_operator(alpha,r) # rotate sample 
    
    r  = stokes.retarder(dlcc)

    p  = stokes.polarizer()
    pt = stokes.rotate_operator(sympy.pi/4,p)   
    
    (pt*r).subs(dlcc,sympy.pi/4)

    
    rs     = stokes.retarder(d_sample)       
    rs_rot = stokes.rotate_operator(alpha,r) # rotate sample 

    
    
    
    # pseudo-inverse
    #N = (M.H * M) ** -1 * M.H
