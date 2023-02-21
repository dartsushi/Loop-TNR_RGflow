import numpy as np
import scipy as scipy
import scipy.linalg as linalg
from scipy.special import iv
import cmath
from scipy import integrate
from scipy.special import eval_legendre
from sympy.physics.wigner import wigner_3j,gaunt

PI=cmath.pi
"""
In this file, I define the local Boltzmann weight for the Ising, Potts, clock, XY, Heisenberg, and 
RP^2 models. (I will update from time to time)
			j
			|
		iー	o ーk
			|
			l
"""


####################   Ising  PART   ####################
def initialize_Ising(T):
    beta = 1./T
    Ising=np.zeros((2,2,2,2))
    c=np.cosh(beta)
    s=np.sinh(beta)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i+j+k+l)==0:
                        Ising[i,j,k,l]=2*c*c
                    elif (i+j+k+l)==2:
                        Ising[i,j,k,l]=2*c*s
                    elif (i+j+k+l)==4:
                        Ising[i,j,k,l]=2*s*s
    return Ising

    ####################   Potts PART   ####################
def initialize_Potts(q,T):
    beta = 1./T
    Potts=np.ones((q,q,q,q))
    bp=np.exp(beta)
    for i in range(q):
        for j in range(q):
            for k in range(q):
                for l in range(q):
                    count=0
                    if i==j:
                        count+=1
                    if j==k:
                        count+=1
                    if k==l:
                        count+=1
                    if l==i:
                        count+=1
                    Potts[i,j,k,l]*=bp**count
    return Potts

####################   Clock   PART   ####################
#q-state clock model. If L=0, it is the usual clock model, where as you can additionally introduce π twist
def initialize_qclock_twist(q,T,L):
    beta=1./T
    clock=np.zeros((q,q,q,q))
    delta=np.zeros((q,q,q,q))
    Q=np.zeros((q,q))
    if L==0:
        for i in range(q):
            for j in range(q):
                Q[i,j]=np.exp(beta*(np.cos(2*PI/q*(i-j))))
    else:
        Q_twist=np.zeros((q,q))
        for i in range(q):
            for j in range(q):
                Q[i,j]=np.exp(beta*(np.cos(2*PI/q*(i-j))))
                Q_twist[i,j]=np.exp(beta*(np.cos(2*PI/q*(i-j)-PI/L)))
    for i in range(q):
        delta[i,i,i,i]=1
    u,s,v=linalg.svd(Q)
    sq_s=np.diag(np.sqrt(s))
    QR=np.einsum("il,lj->ij",u,sq_s)
    QL=np.einsum("is,sj->ij",sq_s,v)
    if L==0:
        clock=np.einsum("mi,nj,ko,lp,mnop->ijkl",QR,QR,QL,QL,delta)
    elif int(np.log2(L))%4==0:
        u_twist,s_twist,v_twist=linalg.svd(Q_twist)
        sq_s_twist=np.diag(np.sqrt(s_twist))
        QR_twist=np.einsum("il,lj->ij",u_twist,sq_s_twist)
        QL_twist=np.einsum("is,sj->ij",sq_s_twist,v_twist)
        clock=np.einsum("mi,nj,ko,lp,mnop->ijkl",QR,QR_twist,QL,QL_twist,delta)
    else:
        u_twist,s_twist,v_twist=linalg.svd(Q_twist)
        sq_s_twist=np.diag(np.sqrt(s_twist))
        QR_twist=np.einsum("il,lj->ij",u_twist,sq_s_twist)
        QL_twist=np.einsum("is,sj->ij",sq_s_twist,v_twist)
        clock=np.einsum("mi,nj,ko,lp,mnop->ijkl",QR_twist,QR,QL_twist,QL,delta)
    return clock

####################   XY  PART   ####################

def initialize_XY(n_max,T):
    beta = 1./T
    XY_1d = np.zeros(2*n_max+1)
    for i in range(-n_max,n_max+1):
        XY_1d[i+n_max] = iv(i,beta)
    XY_1d=np.sqrt(XY_1d)
    XY=np.zeros((2*n_max+1,2*n_max+1,2*n_max+1,2*n_max+1))
    for i in range(2*n_max+1):
        for j in range(2*n_max+1):
            for k in range(2*n_max+1):
                for l in range(2*n_max+1):
                    if (i+j-k-l)==0:
                        XY[i,j,k,l]=XY_1d[i]*XY_1d[j]*XY_1d[k]*XY_1d[l]
    return XY

def initialize_XY_twist(n_max,T,L):
    beta = 1./T
    XY_1d = np.zeros(2*n_max+1)
    for i in range(-n_max,n_max+1):
        XY_1d[i+n_max] = iv(i,beta)
    XY_1d=np.sqrt(XY_1d)
    XY=np.zeros((2*n_max+1,2*n_max+1,2*n_max+1,2*n_max+1),dtype=np.complex)
    for i in range(2*n_max+1):
        for j in range(2*n_max+1):
            for k in range(2*n_max+1):
                for l in range(2*n_max+1):
                    if (i+j-k-l)==0:
                        XY[i,j,k,l]=XY_1d[i]*XY_1d[j]*XY_1d[k]*XY_1d[l]*cmath.exp(PI/L*(k-n_max)*1j)
    return XY
def initialize_XY_twist_y(n_max,T,L):
    beta = 1./T
    XY_1d = np.zeros(2*n_max+1)
    for i in range(-n_max,n_max+1):
        XY_1d[i+n_max] = iv(i,beta)
    XY_1d=np.sqrt(XY_1d)
    XY=np.zeros((2*n_max+1,2*n_max+1,2*n_max+1,2*n_max+1),dtype=np.complex)
    for i in range(2*n_max+1):
        for j in range(2*n_max+1):
            for k in range(2*n_max+1):
                for l in range(2*n_max+1):
                    if (i+j-k-l)==0:
                        XY[i,j,k,l]=XY_1d[i]*XY_1d[j]*XY_1d[k]*XY_1d[l]*cmath.exp(PI/L*(l-n_max)*1j)
    return XY


####################   O^3  PART   #################### 
def f_beta_Heisenberg(l,beta):
    def del_f(x):
        return 2*PI*eval_legendre(l,x)*np.exp(beta*(x-1.))
    return integrate.quad(del_f,-1,1)[0]

def initialize_O3(l_max,T):
    d=l_max+1
    beta=1./T
    coeff1=np.zeros((d**2,d**2,4*d**2))
    coeff2=np.zeros((d**2,d**2,4*d**2))
    for l1 in range(d):
        for l2 in range(d):
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for l in range(abs(l1-l2),l1+l2+1):
                        for m in range(-l,l+1):
                            coeff1[l1**2+l1-m1,l2**2+l2-m2,l**2+l-m]+=gaunt(l1,l2,l,m1,m2,m,prec=11)
                            coeff2[l1**2+l1-m1,l2**2+l2-m2,l**2+l+m]+=gaunt(l1,l2,l,m1,m2,m,prec=11)
    F=np.einsum("ijs,kls->ijkl",coeff1,coeff2)

    fbeta=np.zeros(d)
    for l in range(d):
        fbeta[l]=f_beta_Heisenberg(l,beta)

    Transfer=np.zeros((d**2,d**2,d**2,d**2))
    for l1 in range(d):
        for l2 in range(d):
            for l3 in range(d):
                for l4 in range(d):
                    Value=np.copy(F[l1**2:(l1+1)**2,l2**2:(l2+1)**2,l3**2:(l3+1)**2,l4**2:(l4+1)**2])
                    Value*=(4.*PI)*np.sqrt(fbeta[l1]*fbeta[l2]*fbeta[l3]*fbeta[l4])
                    Transfer[l1**2:(l1+1)**2,l2**2:(l2+1)**2,l3**2:(l3+1)**2,l4**2:(l4+1)**2]=Value
    return Transfer


    ####################   RP^2  PART   #################### 
    
#Lebwohl-Lasher model            
def f_beta_LL(l,beta):
    def del_f(x):
        return 2*PI*eval_legendre(l,x)*np.exp(beta*(x**2-1))
    a=integrate.quad(del_f,-1,1)[0]
    if np.abs(a)<1.e-8:
        return 0
    return a
def index_m(L):
    return int((L+1)*(L/2+1))
def ind(L):
    return slice(index_m(L-2),index_m(L))

def initialize_RP2(l_max,T):
    d=l_max+1
    eff=l_max//2
    beta=1./T
    coeff1=np.zeros((d**2,d**2,4*d**2))
    coeff2=np.zeros((d**2,d**2,4*d**2))
    for l1 in range(d):
        for l2 in range(d):
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for l in range(abs(l1-l2),l1+l2+1):
                        for m in range(-l,l+1):
                            if l1%2==0 and l2%2==0:
                                coeff1[l1**2+l1-m1,l2**2+l2-m2,l**2+l-m]+=gaunt(l1,l2,l,m1,m2,m,prec=14)
                                coeff2[l1**2+l1-m1,l2**2+l2-m2,l**2+l+m]+=gaunt(l1,l2,l,m1,m2,m,prec=14)
    F=np.einsum("ijs,kls->ijkl",coeff1,coeff2)

    fbeta=np.zeros(d)
    for l in range(d):
        fbeta[l]=f_beta_LL(l,beta)
    N_eff=(eff+1)*(2*eff+1)
    Transfer=np.zeros((N_eff,N_eff,N_eff,N_eff))
    for l1 in range(0,2*eff+1,2):
        for l2 in range(0,2*eff+1,2):
            for l3 in range(0,2*eff+1,2):
                for l4 in range(0,2*eff+1,2):
                    Value=np.copy(F[l1**2:(l1+1)**2,l2**2:(l2+1)**2,l3**2:(l3+1)**2,l4**2:(l4+1)**2])
                    Value*=np.sqrt(fbeta[l1]*fbeta[l2]*fbeta[l3]*fbeta[l4])
                    Transfer[ind(l1),ind(l2),ind(l3),ind(l4)]=Value
    return Transfer

#RP^2 gauge model

def f_beta_gauge(l,beta):
    def del_f(x):
        return 4*PI*eval_legendre(l,x)*np.cosh(beta*x)
    a=integrate.quad(del_f,-1,1)[0]
    if np.abs(a)<1.e-6:
        return 0
    return a


def initialize_gauge(l_max,T):
    d=l_max+1
    eff=l_max//2
    beta=1./T
    coeff1=np.zeros((d**2,d**2,4*d**2))
    coeff2=np.zeros((d**2,d**2,4*d**2))
    for l1 in range(d):
        for l2 in range(d):
            for m1 in range(-l1,l1+1):
                for m2 in range(-l2,l2+1):
                    for l in range(abs(l1-l2),l1+l2+1):
                        for m in range(-l,l+1):
                            if l1%2==0 and l2%2==0:
                                coeff1[l1**2+l1-m1,l2**2+l2-m2,l**2+l-m]+=gaunt(l1,l2,l,m1,m2,m,prec=14)
                                coeff2[l1**2+l1-m1,l2**2+l2-m2,l**2+l+m]+=gaunt(l1,l2,l,m1,m2,m,prec=14)
    F=np.einsum("ijs,kls->ijkl",coeff1,coeff2)

    fbeta=np.zeros(d)
    for l in range(d):
        fbeta[l]=f_beta_gauge(l,beta)
    N_eff=(eff+1)*(2*eff+1)
    Transfer=np.zeros((N_eff,N_eff,N_eff,N_eff))
    for l1 in range(0,2*eff+1,2):
        for l2 in range(0,2*eff+1,2):
            for l3 in range(0,2*eff+1,2):
                for l4 in range(0,2*eff+1,2):
                    Value=np.copy(F[l1**2:(l1+1)**2,l2**2:(l2+1)**2,l3**2:(l3+1)**2,l4**2:(l4+1)**2])
                    Value*=np.sqrt(fbeta[l1]*fbeta[l2]*fbeta[l3]*fbeta[l4])
                    Transfer[ind(l1),ind(l2),ind(l3),ind(l4)]=Value
    return Transfer