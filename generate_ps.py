import os
import numpy as np
import multiprocessing
import camb
from camb import model, initialpower

cmb_para=np.load('./data/camb_new_processed/parameters.npy')
n_sample=cmb_para.shape[0]

def cal_camb(i):
    camb_TT_new=np.zeros((2499+6)) # l={2,3,...2500}, n_param = 6
    camb_TE_new=np.zeros((2499+6))
    camb_EE_new=np.zeros((2499+6))
    
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=cmb_para[i,5]*100, ombh2=cmb_para[i,0], omch2=cmb_para[i,1], mnu=0.06, omk=0, tau=cmb_para[i,2])
    pars.InitPower.set_params(As=cmb_para[i,3], ns=cmb_para[i,4], r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
    unlensedCL=powers['unlensed_scalar']
    
    camb_TT_new[6:]=unlensedCL[2:2501,0] # TT
    camb_TE_new[6:]=unlensedCL[2:2501,3] # TE
    camb_EE_new[6:]=unlensedCL[2:2501,1] # EE
    
    camb_TT_new[:6]=cmb_para[i,:]
    camb_TE_new[:6]=cmb_para[i,:]
    camb_EE_new[:6]=cmb_para[i,:]

    np.save(f'./data/camb_new/camb_TT_{i}.npy', camb_TT_new)
    np.save(f'./data/camb_new/camb_TE_{i}.npy', camb_TE_new)
    np.save(f'./data/camb_new/camb_EE_{i}.npy', camb_EE_new)

for i in range(10000,n_sample):
    cal_camb(i)
#if __name__ == '__main__':
#    cores = multiprocessing.cpu_count()
#    print (cores)
#    pool = multiprocessing.Pool(processes=20)
#    num = range(10000)
#    pool.map(cal_camb, num)
