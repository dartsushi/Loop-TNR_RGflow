# S.Yang et al., "Loop Optimization for Tensor Network Renormalization" (2017)
import numpy as np
import entanglement_filtering as ef
from copy import copy
from ncon import ncon

""" Four leg tensor A_{ijkl}

   j
   |
i--A--k    
   |
   l

   |
--S1
    \   
     S2--
     |    

"""

def SVD12(A, d_cut):
    shape = A.shape
    chi = min(d_cut, shape[0]*shape[1])
    mat = A.reshape(shape[0]*shape[1],-1)
    u,s,vh = np.linalg.svd(mat)
    sq_s = np.diag(np.sqrt(s[:chi]))
    s1 = np.dot(u[:,:chi],sq_s).reshape(shape[0],shape[1],-1)
    s2 = np.dot(sq_s,vh[:chi,:]).reshape(-1,shape[2],shape[3])
    return s1, s2

#Create a 2x2 transfer matrix in y-direction. 
def transfer_matrix(tA, tB):
    row=np.einsum("ijkl,kmin->jmln",tA,tB)
    row=np.einsum("ijkl,lknm->ijmn",row,row)
    return row

# Normalization of the tensors and their norm 
def normalize_tensor(tA, tB):
    norm = transfer_matrix(tA, tB)
    norm = np.trace(norm, axis1=0, axis2=2)
    norm = np.trace(norm, axis1=0, axis2=1)
    sitenorm = norm**(1/4)
    return tA / sitenorm, tB / sitenorm, norm
"""

       |   |			  |   |   |   |
	 --s1--s2--   	->  --s1--PR--PL--s2--

"""
def one_loop_projector(phi, pos, d_cut):
    # Initializaiton with one-loop 
	# L. Wang and F. Verstraete arXiv:1110.4362
    L = np.identity(np.shape(phi[0])[0], dtype=complex)
    for n in range(pos + 1):
        L = ef.QR_L(L, phi[n])
    R = np.identity(np.shape(phi[-1])[-1], dtype=complex)
    for n in range(len(phi), pos + 1, -1):
        R = ef.QR_R(R, phi[n - 1])
    args = {'bondm': d_cut}
    return ef.P_decomposition(R,L,args,mode='bondm')

def make_psiB(tA, tB, d_cut):
    psiA = ef.make_psi(tA, tB, reshape=False)
    psi = ef.make_psi(tA, tB)
    psiB = []
    # filter phi for one-loop and insert the projector
    for i in range(4):
        s1, s2 = SVD12(psiA[i], 2*d_cut)
        phi = copy(psi)
        del phi[i]
        phi.insert(i, s1)
        phi.insert(i + 1, s2)
        PR, PL = one_loop_projector(phi, i, d_cut)
        psiB.append(np.tensordot(s1, PR, axes=1))
        psiB.append(np.tensordot(PL, s2, axes=1))
    return psiB

"""
--[ psi*[i]]--
   |   |
--[ psi[i] ]--
"""

# f(T) = C+T†NT-W†T-T†W
def cost_func(i, psiA, psiB):
    C = const_C(psiA)
    N = tN(i, psiB)
    TNT = ncon((np.conj(psiB[i]), N, psiB[i]), ([1,3,2],[2,1,5,4],[4,3,5]))
    W = tW(i, psiA, psiB)
    WdT = ncon((psiB[i], np.conj(W)),([1,2,3],[3,1,2]))
    TdW = ncon((np.conj(psiB[i]), W),([1,2,3],[3,1,2]))
    return C + TNT - WdT - TdW

def const_C(psiA):
	tmp = ncon((psiA[0], np.conj(psiA[0])), ([-1, 1, 2, -3], [-2, 1, 2, -4]))
	for i in range(1, 4, 1):
		tmp = ncon((tmp, psiA[i], np.conj(psiA[i])), ([-1, -2, 1, 2], [1, 3, 4, -3], [2, 3, 4, -4]))
	return np.einsum("ijij",tmp)

def tN(pos, psiB):
	tmp = ncon((psiB[(pos+1)%8], np.conj(psiB[(pos+1)%8])),([-3,1,-4],[-1,1,-2]))
	for j in range(pos+2, pos+8):
		tA = ncon((np.conj(psiB[j%8]), psiB[j%8]), ([-1,1,-2],[-3,1,-4]))
		tmp = ncon((tmp,tA),([-1,1,-3,2],[1,-2,2,-4]))
	return tmp


def tW(pos, psiA, psiB):
    pos_psiA = pos // 2
    j = (pos_psiA+1)%4
    W = ncon((np.conj(psiB[2*j]), np.conj(psiB[2*j+1]), psiA[j]),([-1,1,2],[2,3,-2],[-3,1,3,-4]))
    for p in range(pos_psiA + 2, pos_psiA + 4):
        j = p % 4
        psiABB = ncon((np.conj(psiB[2*j]), np.conj(psiB[2*j+1]), psiA[j]),([-1,1,2],[2,3,-2],[-3,1,3,-4]))
        W = ncon((W, psiABB),([-1,1,-3,2],[1,-2,2,-4]))
    if pos % 2 == 0:
        return ncon((W, np.conj(psiB[pos+1]),psiA[pos_psiA]),([1,-2,2,3],[-1,4,1],[3,-3,4,2]))
    else:
        return ncon((W, np.conj(psiB[pos-1]),psiA[pos_psiA]),([-1,1,2,3],[1,4,-2],[3,4,-3,2]))

"""
[N_i]
--i[  ]j--
--k[  ]l--

[W_i]
j--[   ]--i
     |
	 k
"""
# N_iT_i = W_i
def optimize_Tn(N, W):
    mat_N = N.reshape((N.shape[0]*N.shape[1], -1))
    mat_W = W.reshape((-1, W.shape[2]))
    # rcond can be changed!(hyper parameter.... lower -> better)
    Tn = np.linalg.lstsq(mat_N, mat_W, rcond=1.0e-12)[0]
    Tn = Tn.reshape((N.shape[2], N.shape[3], W.shape[2]))
    Tn = np.transpose(Tn, axes=(1,2,0))
    return Tn

"""
[A']		[B']
i	  j		i	  j
\	 /		\	 /
 1--6   	 4--3
 |  |		 |  |
 2--5		 7--0
/    \		/	 \	
l     k    l	  k
"""
def loop_opt(tA, tB, d_cut=16, error_limit=1.0E-14, maxloop=50):
    psiA = ef.make_psi(tA, tB, reshape=False)
    psiB = make_psiB(tA, tB, d_cut)
    loop_counter = 0
    cost = np.inf
    while ((np.abs(cost) > error_limit) and (loop_counter < maxloop)):
        for n in range(8):
            psiB[n] = optimize_Tn(tN(n, psiB), tW(n, psiA, psiB))
        loop_counter += 1
        cost = cost_func(0, psiA, psiB)
    tA = ncon((psiB[1],psiB[6],psiB[5],psiB[2]),([-1,1,2],[3,1,-2],[-3,4,3],[2,4,-4]))
    tB = ncon((psiB[4],psiB[3],psiB[0],psiB[7]),([1,2,-1],[-2,3,1],[4,3,-3],[-4,2,4]))
    return tA, tB, cost

def TNR_step(tA, tB, d_cut, error_limit, maxloop):
    tA, tB = ef.filter(tA, tB)
    tA, tB, _cost_err = loop_opt(tA, tB, d_cut, error_limit, maxloop)
    tA, tB, plqnorm = normalize_tensor(tA, tB)
    return tA, tB, plqnorm
