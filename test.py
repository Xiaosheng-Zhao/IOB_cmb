import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # depend on the GPU devices

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import torch.optim as optim
from torch.utils.data import random_split
from pytorchtools import EarlyStopping
from random import randint, random
from torch.utils.data import Dataset
from torch import nn
from model_cmb import Encoder_cmb, Decoder_cmb, maskin_cmb
from load_data_cmb import CMBDataset
import argparse

# The two functions to return to the original data space
def trans(data,min_temp,data_min,data_max):
    return 10**(data*(data_max-data_min)+data_min)-1+min_temp

def trans_error(data,error,min_temp,data_min,data_max):
    return error*((data+1-min_temp)*np.log(10))**2*(data_max-data_min)**2

### Testing function for the retrained model
def test_epoch_original(encoder, decoder, maskin, d_encode, device, dataloader, loss_fn,maskindex,min_temp,data_min,data_max):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = [] # For the encoded latents
        conc_param = []
        conc_out_encoded = []
        
        conc_data = [] # For the real input data
        conc_data_mean = []
        conc_err = []
        
        # Data untranslated (by the scaling above)
        conc_data_untrans = [] 
        conc_data_mean_untrans = []
        conc_err_untrans = []
        
        # Mask
        range_index=torch.arange(1, d_encode+1)
        mask = range_index.le(maskindex).unsqueeze(0).to(device)

        for data_batch, data_batch_mean, param,err in dataloader:
            data_batch = data_batch.to(device)
            data_batch_mean = data_batch_mean.to(device)
            param = param.to(device)
            err = err.to(device)
            # Encode data
            encoded_data= encoder(param)
            maskin_data = maskin(encoded_data)

            decoded_data = decoder(maskin_data*mask)

            conc_out_encoded.append(maskin_data.cpu())
            conc_param.append(param.cpu())
            
            conc_out.append(trans(decoded_data.cpu(),min_temp,data_min,data_max))
            conc_data.append(trans(data_batch.cpu()[:,0,:],min_temp,data_min,data_max))
            conc_data_mean.append(trans(data_batch_mean.cpu()[:,0,:],min_temp,data_min,data_max))
            conc_err.append(trans_error(trans(data_batch.cpu()[:,0,:],min_temp,data_min,data_max),err.cpu(),min_temp,data_min,data_max))
            
            conc_data_untrans.append(data_batch.cpu())
            conc_data_mean_untrans.append(data_batch_mean.cpu())
            conc_err_untrans.append(err.cpu())
            
        # Create a single tensor with all the values in the lists
        conc_out_encoded = torch.cat(conc_out_encoded)
        conc_param = torch.cat(conc_param)
        
        conc_out = torch.cat(conc_out)
        conc_data = torch.cat(conc_data)
        conc_data_mean = torch.cat(conc_data_mean)
        conc_err = torch.cat(conc_err)
        
        conc_data_untrans = torch.cat(conc_data_untrans)
        conc_data_mean_untrans = torch.cat(conc_data_mean_untrans)
        conc_err_untrans = torch.cat(conc_err_untrans)
        
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_data_mean[:,:],conc_err)
    return val_loss.data, conc_out_encoded, conc_param,conc_data_mean_untrans[:,0,:], conc_err_untrans, conc_data_untrans

def loss_fn(output, target, weight):
    loss = torch.mean((output - target)**2/weight)
    return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="The basic name of encoder-decoder experiments",
                        type=str,
                        default='shallow96-test',
                        required=False)
    
    parser.add_argument("--cmb_type",
                        help="TT,TE,or EE",
                        type=str,
                        default='TT',
                        required=False)

    
    args = parser.parse_args()
    ############################################################### Key parameters to vary
    cmb_type=args.cmb_type # CMB type
    name=cmb_type+'_'+args.model_name # The basic name of this experiments
    d_encode=6 #128, the bottleneck dimension
    ################################################################

    ### Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')
    
    ### Prepare the data
    data_dir = './data/camb_new_processed/d_normed_new_noised_%s.npy'%cmb_type
    data_mean_dir = './data/camb_new_processed/d_normed_new_noised_mean_%s.npy'%cmb_type
    para_dir = './data/camb_new_processed/p_normed_new_noised_%s.npy'%cmb_type
    err_dir = './data/camb_new_processed/e_normed_new_noised_%s.npy'%cmb_type
    
    batch_size = 128
    dataset = CMBDataset(data_dir,data_mean_dir,para_dir,err_dir)
    
    torch.manual_seed(43)
    val_size = 500 
    test_size = 500
    train_size = len(dataset) - val_size - test_size
    X_train, X_valid, X_test= random_split(dataset, [train_size, val_size, test_size])
    
    # data loader setup
    train_loader = torch.utils.data.DataLoader(X_train,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device == device))
    valid_loader = torch.utils.data.DataLoader(X_valid,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device == device))
    test_loader = torch.utils.data.DataLoader(X_test,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=(device == device))
    
    # Inverse process of generating the rescaled data
    # Basically only need ``min_temp, data_min, and data_max'', in order to 
    # recover the data in the original space from rescaled ones
    
    min_temp = np.load('./data/camb_new_processed/min_temp_%s.npy'%cmb_type)
    data_min = np.load('./data/camb_new_processed/data_min_%s.npy'%cmb_type)
    data_max = np.load('./data/camb_new_processed/data_max_%s.npy'%cmb_type)
    
    ### Load only the last model, and mask the bottleneck gradually
    test_losses_onloader_onemodel=[]
    
    encoder = Encoder_cmb(encoded_space_dim=d_encode)
    decoder = Decoder_cmb(encoded_space_dim=d_encode)
    maskin = maskin_cmb()
    
    model_state_save_early_path_checkpoint ='./model/checkpoint_%s.tar'%(name)
    checkpoint_model = torch.load(model_state_save_early_path_checkpoint)
    encoder.load_state_dict(checkpoint_model['encoder_state_dict'])
    decoder.load_state_dict(checkpoint_model['decoder_state_dict'])
    maskin.load_state_dict(checkpoint_model['maskin_state_dict'])
    
    encoder.eval()
    encoder.to(device)
    decoder.eval()
    decoder.to(device)
    maskin.eval()
    maskin.to(device)
    
    for maskindex in range(1,d_encode+1):
        if maskindex==d_encode:
            temp, conc_out_encoded, conc_param, conc_data_mean, conc_err, conc_data=test_epoch_original(encoder, decoder, maskin, d_encode, device, test_loader, loss_fn,maskindex,min_temp,data_min,data_max) # Here we did not use the test_loader, but load the test data from scratch
        else:
            temp, _,_,_,_,_=test_epoch_original(encoder, decoder, maskin, d_encode, device, test_loader, loss_fn,maskindex,min_temp,data_min,data_max) # Here we did not use the test_loader, but load the test data from scratch
    
        test_losses_onloader_onemodel.append(np.array(temp))
    
    # Save the testing samples for operon use later
    np.save('./data/camb_new_processed/param_%s.npy'%name,conc_param)
    np.save('./data/camb_new_processed/encoded_%s.npy'%name,conc_out_encoded)
    np.save('./data/camb_new_processed/data_%s.npy'%name,conc_data)
    np.save('./data/camb_new_processed/data_mean_%s.npy'%name,conc_data_mean)
    np.save('./data/camb_new_processed/error_%s.npy'%name,conc_err)
    
    print ("encoded_shape:",conc_out_encoded.shape)
    print ("Weighted loss value:",np.array(test_losses_onloader_onemodel))
    
    ### Plot the weighted loss
    fig,ax = plt.subplots(figsize=(8,6))
    ax.plot(np.arange(1,len(test_losses_onloader_onemodel)+1),test_losses_onloader_onemodel)
    ax.set_xlabel('# of effective channels',fontsize=16)
    ax.set_ylabel('Weighted MSE',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.axhline(1., color="grey", linestyle="--") # The threshold for the weighted error (likelihood)
    plt.show()
    plt.close()    

if __name__ == '__main__':
    main()