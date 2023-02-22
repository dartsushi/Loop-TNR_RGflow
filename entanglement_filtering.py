# S.Yang et al., "Loop Optimization for Tensor Network Renormalization" (2017)
import numpy as np
from ncon import ncon


"""
[Tensor_list]
entanglement_filtering.make_psi
-> tA
-> tB.transpose(1, 2, 3, 0)
-> tA.transpose(2, 3, 0, 1)
-> tB.transpose(3, 0, 1, 2)
Tensor_list transposes the tensors so that SVD_tensor properly decompose the square ABAB to the octagon.
	 8	   1
	 |	2' |				  j		 i		l	   k
 7---B-----A---2			  |	     | 	    |  	   |
  3' |	   |1'			--> i-A-k  l-B-j  k-A-i  j-B-l	
	 |	4' |				  |	     | 	    |  	   |
 6---A-----B---3			  l		 k		j	   i
	 |     |
	 5     4


psiA = make_psi(tA, tB, original_shape = True)
len(psiA) = 4
len(psiB) = 8
psiA is a list of the 4-leg tensors.
psiB is a list of the 3-leg tensors.
							  [PLR]
      |	  ->   |   ->	  | ->  ↓ <-|	<-       |	  <-
-L---psi[0]---psi[1]--…-psi[2*i]---psi[2*i+1]-…psi[len(psi)-1]-R-

"""

# reshape the tensors to 3-leg tensors if "reshape == True"
def make_psi(tA, tB, reshape = True):
    psi = [tA]
    psi.append(np.transpose(tB, axes=(1,2,3,0)))
    psi.append(np.transpose(tA, axes=(2,3,0,1)))
    psi.append(np.transpose(tB, axes=(3,0,1,2)))
    # conbine the two physical legs
    if reshape == True:
        for n in range(len(psi)):
            shape = np.shape(psi[n])
            psi[n] = np.reshape(psi[n], (shape[0], shape[1]*shape[2], shape[3]))
    return psi

def svd_truncate(u, s, vh, args, mode='value'):
    if mode == 'value':
        chi = len(np.where(s[:] > args['cutoff'])[0])
    elif mode == 'bondm':
        chi = min(args['bondm'], len(s))
    s = s[:chi]
    u = u[:, :chi]
    vh = vh[:chi, :]
    return u, s, vh


"""
   two indices combined
   	 |		   ---             |    
--L--Tn---- -> ---LTn--- ->  --Tn--Lnew--  
"""
def QR_L(L, Tn):
    LTn = np.tensordot(L, Tn, axes=1)
    L_new = np.linalg.qr(LTn.reshape(-1, LTn.shape[2]), mode='r')
    return L_new

"""
   two indices combined
   |		        ---          |    
--Tn--R--- -> ---TnR--- ->  --R--Tn--  
"""
def QR_R(R, Tn):
    TnR = np.tensordot(Tn, R, axes=1)
    R_new = np.linalg.qr(np.transpose(TnR.reshape(TnR.shape[0],-1)), mode='r')
    return np.transpose(R_new)


"""

---Ln--Tn---

"""
def find_L(psi, pos, error_limit=1.0e-12, maxloop=50):
    old_L = np.identity(psi[pos].shape[0], dtype=complex)
    loop_counter = 0
    error = np.inf

    while error > error_limit and loop_counter < maxloop:
        new_L = np.copy(old_L)
        for i in range(pos, pos + 4):
            new_L = QR_L(new_L, psi[i % 4])
        new_L /= np.max(np.abs(new_L))
        if old_L.shape == new_L.shape:
            error = np.linalg.norm(new_L - old_L)
        old_L = np.copy(new_L)
        loop_counter += 1
    return old_L

"""

---T(n-1)--Rn---

"""
def find_R(psi, pos, error_limit=1.0e-12, maxloop=50):
    old_R = np.identity(psi[pos-1].shape[2], dtype=complex)
    loop_counter = 0
    error = np.inf

    while error > error_limit and loop_counter < maxloop:
        new_R = np.copy(old_R)
        for i in np.arange(pos - 1, pos - 1 - 4, -1):
            new_R = QR_R(new_R, psi[i % 4])
        new_R /= np.max(np.abs(new_R))
        if old_R.shape == new_R.shape:
            error = np.linalg.norm(new_R - old_R)
        old_R = np.copy(new_R)
        loop_counter += 1
    return old_R


def P_decomposition(R, L, args, mode='value'):
    temp = np.dot(L, R)
    u, s, vh = np.linalg.svd(temp, full_matrices=False)
    # discard singular values smaller than 1.0e-12
    u, s, vh = svd_truncate(u, s, vh, args, mode)
    re_sq = np.diag(1/np.sqrt(s))
    # P_R(n) = R(n) dag(Vh(n)) 1/sqrt(S(n))
    PR = np.dot(R, np.transpose(np.conj(vh)))
    PR = np.dot(PR, re_sq)
    # P_L(n) = 1/sqrt(S(n)) dag(U(n)) L(n)
    PL = np.dot(re_sq, np.transpose(np.conj(u)))
    PL = np.dot(PL, L)
    return PR, PL
"""
	 	   
	 |	       |
  ---B-PR0-PL0-A---
	 |	   	   |
	PL3		  PR1  
	 |	       |
	PR3		  PL1
	 |		   |
  ---A-PL2-PR2-B---
	 |         |
		  

"""
def find_projector(psi, maxloop=50):
    args = {'cutoff': 1.0e-12, 'bondm': 1024}
    PR_list = []
    PL_list = []

    for n in range(4):
        R = find_R(psi, n, 1.0e-12, maxloop)
        L = find_L(psi, n, 1.0e-12, maxloop)
        PR, PL = P_decomposition(R, L, args)
        PR_list.append(PR)
        PL_list.append(PL)
    return PR_list, PL_list

"""
The order of the indices 
			j
			|
		iー	o ーk
			|
			l

Redefinition of tA and tB. Note that PnR = PR_list[n - 1] and PnL = PL_list[n - 1]
The arrows the in-out direction of the matrices.
[tA'] 			[tB']
	  |				  |
	 PR3			 PL1
	  Λ 			  v
	  |				  |
-PL0->tA<-PL2- 	-PR2-<tB>-PR0-
	  |				  |
	  v				  Λ
	 PR1			 PL3
	  |				  |

"""

def filter(tA, tB):
    psi = make_psi(tA,tB)
    PR_list, PL_list = find_projector(psi)
    tA = ncon((tA,PL_list[0],PR_list[3],PL_list[2],PR_list[1]),([1,2,3,4],[-1,1],[2,-2],[-3,3],[4,-4]))
    tB = ncon((tB,PR_list[2],PL_list[1],PR_list[0],PL_list[3]),([1,2,3,4],[1,-1],[-2,2],[3,-3],[-4,4]))
    return tA, tB