import numpy as np
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmb_type",
                        help="TT,TE,or EE",
                        type=str,
                        default='TT',
                        required=False)
    
    args = parser.parse_args()
    ############################################################### Key parameters to vary
    cmb_type=args.cmb_type # CMB type
    ################################################################
    
    # Load data from scratch
    files=glob.glob('./data/camb_new/*%s*npy'%cmb_type)
    data=[]
    for i in range(len(files)):
        d=np.load(files[i])
        data.append(d)
    data=np.array(data)
    l=np.arange(2,2501)
    
    error=(2/(2*l+1))*data[:,6:]**2
    
    np.random.seed(0)
    data_noise = np.random.normal(0,np.sqrt(error))+data[:,6:]
    min_temp = np.min(data_noise)
    data_noise = data_noise - min_temp + 1 # Ensure a positive array
    
    
    # The rescaled powerspectra
    para_min=np.min(data[:,:6],axis=0)
    para_max=np.max(data[:,:6],axis=0)
    para_norm=(data[:,:6]-para_min)/(para_max-para_min)
    
    data_min=np.min(np.log10(data_noise),axis=0)
    data_max=np.max(np.log10(data_noise),axis=0)
    
    data_norm=(np.log10(data_noise)-data_min)/(data_max-data_min)
    err_norm=error/(data_max-data_min)**2/(data_noise*np.log(10))**2
    
    np.save('./data/camb_new_processed/p_normed_new_noised_%s.npy'%cmb_type,para_norm)
    np.save('./data/camb_new_processed/d_normed_new_noised_%s.npy'%cmb_type,data_norm)
    np.save('./data/camb_new_processed/e_normed_new_noised_%s.npy'%cmb_type,err_norm)
    
    
    np.save('./data/camb_new_processed/min_temp_%s.npy'%cmb_type,min_temp)
    np.save('./data/camb_new_processed/data_min_%s.npy'%cmb_type,data_min)
    np.save('./data/camb_new_processed/data_max_%s.npy'%cmb_type,data_max)
    
    # The rescaled mean of the powerspectra
    data_noise_mean = data[:,6:]
    data_noise_mean = data_noise_mean - min_temp + 1
    
    # Save the mean spectra
    data_norm_mean=(np.log10(data_noise_mean)-data_min)/(data_max-data_min)
    np.save('./data/camb_new_processed/d_normed_new_noised_mean_%s.npy'%cmb_type,data_norm_mean)

if __name__ == '__main__':
    main()