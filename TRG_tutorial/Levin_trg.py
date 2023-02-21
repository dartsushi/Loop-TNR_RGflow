import numpy as np
from scipy.linalg import svd
from ncon import ncon
from tqdm import tqdm


""" Four leg tensor A_{ijkl}

   j
   |
i--A--k    
   |
   l


   \      /
    S1--S2
    |    |
    S4--S3
   /      \

"""

def SVD13(A, chi):
    shape = A.shape
    chi_max = min(chi, shape[0]*shape[1])
    mat = A.transpose(2,3,0,1).reshape(shape[2]*shape[3],-1)
    u,s,v = np.linalg.svd(mat)
    sq_s = np.diag(np.sqrt(s[:chi_max]))
    S1 = np.dot(u[:,:chi_max],sq_s).reshape(shape[2],shape[3],-1)
    S3 = np.dot(sq_s,v[:chi_max,:]).reshape(-1,shape[0],shape[1])
    return S1,S3

def SVD24(A, chi):
    shape = A.shape
    chi_max = min(chi, shape[0]*shape[3])
    mat = A.transpose(3,0,1,2).reshape(shape[3]*shape[0],-1)
    u,s,v = np.linalg.svd(mat)
    sq_s = np.diag(np.sqrt(s[:chi_max]))
    S2 = np.dot(u[:,:chi_max],sq_s).reshape(shape[3],shape[0],-1)
    S4 = np.dot(sq_s,v[:chi_max,:]).reshape(-1,shape[1],shape[2])
    return S2,S4


def contract_S(S1, S2, S3, S4):
    return ncon([S1,S2,S3,S4],[[1,4,-1],[2,1,-2],[-3,3,2],[-4,4,3]])

def TRG_step(A, chi):
    S1,S3 = SVD13(A,chi)
    S2,S4 = SVD24(A,chi)
    return contract_S(S1,S2,S3,S4)

def normalize_tensor(A):
    norm = np.einsum("ijij",A)
    return A/norm,norm

def TRG_lnz(initial_A, chi, RG_step):
    A,factor = normalize_tensor(initial_A)
    lnz = np.log(factor)
    for i in tqdm(range(RG_step)):
        A = TRG_step(A,chi)
        A,factor = normalize_tensor(A)
        lnz += np.log(factor)/2**(i+1)
    return lnz
