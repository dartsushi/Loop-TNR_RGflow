# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    cft.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Atsushi <aueda@issp.u-tokyo.ac.jp>         +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/03/21 23:29:17 by Atsushi           #+#    #+#              #
#    Updated: 2023/02/21 17:44:03 by Atsushi          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
from loop_opt import transfer_matrix

#Calculation of the scaling dimensions and central charge 
def cal_sdimension(tA, tB, plqnorm):
    tm = transfer_matrix(tA, tB)
    tm = tm.reshape(tm.shape[0]*tm.shape[1], tm.shape[2]*tm.shape[3])
    dim = tm.shape[0]

    # cft_data[0]->central charge, cft_data[n]->scaling dimension 
    # Note this is the case for x_0 = 0, which is not true for the 
    # twisted boundary condition of the XY model 
    cft_data = np.zeros(np.shape(tm)[0])
    eigvals = np.linalg.eigvals(tm)
    eig_real = np.sort(np.real(eigvals))[::-1]

    # ZC. Gu et al., Phys. Rev. B 80, 155131 (2009)
    cft_data[0] = 12. * np.log(np.abs(eig_real[0] / plqnorm)) / 2. / np.pi
    for i in range(1, dim):
        cft_data[i] = np.log(np.abs(eig_real[0] / eig_real[i])) / 2. / np.pi 
    return cft_data