import numpy as np
from Initialization import initialize_Ising
import entanglement_filtering as ef
import loop_opt as tnr

import cft
from tqdm import tqdm
Ising_Tc = 2./np.log(1.+np.sqrt(2))

def collect_Ising(T, d_cut, size):
    keep_num_scaling = d_cut ** 2
    error_limit = 1.0E-14
    maxloop = 50
    area = 4.

    cft_result = np.zeros((size, keep_num_scaling))
    norm_list = []

    tA = initialize_Ising(T)
    tB = np.copy(tA)

    tA, tB, plqnorm = tnr.normalize_tensor(tA, tB)
    norm_list.append(plqnorm)
    ln_z = np.log(plqnorm)/area

    for i in tqdm(range(size)):
        tA, tB, plqnorm = tnr.TNR_step(tA,tB,d_cut,error_limit,maxloop)
        area *= 2
        ln_z += np.log(plqnorm) / area
        norm_list.append(plqnorm)
        data = cft.cal_sdimension(tA, tB, plqnorm)
        cft_result[i, 0:min(keep_num_scaling, np.shape(data)[0])] = data[0:min(keep_num_scaling, np.shape(data)[0])]
    print("Central charge:")
    print(cft_result[:,0])
    print("Scaling dimension of the magnetic operator")
    print(cft_result[:,1])

collect_Ising(Ising_Tc, 8, 12)