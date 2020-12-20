def conjgrad(Af,b,x0,itmax=100):
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
        r = r-alpha*Ap;
        rsnew = r.T.dot(r);
        if np.sqrt(rsnew).sum()<1e-10:
            break;
        p=r+(rsnew/rsold)*p;
        rsold=rsnew;
        res.append(np.sqrt((r**2).sum()))
    return x,res

def solveThikonov(forward,adjoint,b,mu,x0,itmax=100):
    # b - measurement
    forward_Tikh = lambda x: (adjoint(forward(x))) + mu*x 
    b_Tikh = adjoint(b)
    w0,res = conjgrad(forward_Tikh,b_Tikh,x0,itmax=itmax)    
    return w0, res
    
