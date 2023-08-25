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

# Define functions to print expressions that are easy to read
from IPython.display import display
import sympy as sp
from sympy import *

import pandas as pd
import csv


def pprint(expr):
    # Define the variable for pprint
    X6= Symbol('X6')
    X1 = Symbol('X1')
    X2 = Symbol('X2')
    X3 = Symbol('X3')
    X4 = Symbol('X4')
    X5 = Symbol('X5')
    
    display(eval(expr))
    
# The two functions to return to the original data space
def trans(data,min_temp,data_min,data_max):
    return 10**(data*(data_max-data_min)+data_min)-1+min_temp

def trans_error(data,error,min_temp,data_min,data_max):
    return error*((data+1-min_temp)*np.log(10))**2*(data_max-data_min)**2


### Testing function for the retrained model
def test_epoch_replace(encoder, decoder, maskin, d_encode,device, dataloader, loss_fn,maskindex, ind, ind_var,min_temp,data_min,data_max,name,sr_pred):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Define the lists to store the outputs for (1) batch of 500 testing samples
        conc_out = [] 
        conc_data_mean = [] 
        conc_err = []
        
        # Mask
        range_index=torch.arange(1, d_encode+1)
        mask = range_index.le(maskindex).unsqueeze(0).to(device)

        count = 0

        param = torch.tensor(np.float32(np.load('./data/camb_new_processed/param_%s.npy'%name)))
        data_batch_mean = torch.tensor(np.float32(np.load('./data/camb_new_processed/data_mean_%s.npy'%name)))
        err = torch.tensor(np.float32(np.load('./data/camb_new_processed/error_%s.npy'%name)))
        data_batch = torch.tensor(np.float32(np.load('./data/camb_new_processed/data_%s.npy'%name)))
            
        data_batch = data_batch.to(device)
        param = param.to(device)
        err = err.to(device)
        #data = data.to(device)
        
        # Encode data
        encoded_data= encoder(param)
        maskin_data = maskin(encoded_data)
        
        # Replacement
        sr_pred_i = sr_pred[ind] # Dimension: each ind has 500, the # of the testing samples
        maskin_data[:,ind_var] = sr_pred_i.to(device)
        
        # Decode data
        decoded_data = decoder(maskin_data*mask)
        
        conc_out.append(trans(decoded_data.cpu(),min_temp,data_min,data_max))
        conc_data_mean.append(trans(data_batch_mean[:,:],min_temp,data_min,data_max))
        conc_err.append(trans_error(trans(data_batch.cpu()[:,:],min_temp,data_min,data_max),err.cpu(),min_temp,data_min,data_max))
                               
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_data_mean = torch.cat(conc_data_mean)
        conc_err = torch.cat(conc_err)

        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_data_mean[:,:],conc_err)
    return val_loss.data

### Define the loss function
#loss_fn = torch.nn.MSELoss()
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

    parser.add_argument("--operon_name",
                        help="The basic name of the operon experiments",
                        type=str,
                        default='shallow96-test',
                        required=False)
    
    parser.add_argument("--thin",
                        help="thin the 2000 individual expressions for each latent by an interval",
                        type=int,
                        default=200,
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
    operon_name = cmb_type+'_'+args.operon_name # The basic name for the operon
    d_encode=6 #128, the bottleneck dimension
    interval = args.thin
    torch.manual_seed(43)
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
    
    
    # For the pareto
    print('Outputting good expressions of the first latent with final weighted mse less than 1 for pareto')
    with open('./data/sr/pareto_good_model_latent%d_model_%s_operon_%s.csv'%(1, name,operon_name), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['length','weighted MSE','expression'])
        
        maskindex = 6
        for ind_var in range(maskindex):
            test_name = operon_name+'_latent%d'%ind_var # index of the latent variable
            test_losses_onloader_onemodel_parato=[]
            
            # Symbolic regression (sr) predictions
            # sr prediction: [len(expressions),500]
            sr_pred = torch.tensor(np.float32(np.load('./data/sr/pred_pareto_%s.npy'%test_name))) 
            # mse: [len(expressions),]
            mse_pred = np.load('./data/sr/mse_pareto_%s.npy'%test_name) 
            # expressions: [len(expressions),]
            data0=pd.read_csv('./data/sr/pareto_%s.csv'%test_name) 
            # Complexity of the expressions
            complexity=pd.read_csv('./data/sr/pareto_%s.csv'%test_name)['length'].values
        
            # Lengh of the points in the parato front
            num_plot = len(sr_pred)
            
            for ind in range(num_plot):
                temp=test_epoch_replace(encoder, decoder, maskin, d_encode,device, test_loader, loss_fn,maskindex, ind, ind_var,min_temp,data_min,data_max,name,sr_pred) # Here we did not use the test_loader, but load the test data from scratch
                test_losses_onloader_onemodel_parato.append(np.array(temp))
                if ind_var==0 and temp<=1:
                #if True:
                    writer.writerow([complexity[ind],np.round(np.array(temp),3),data0['infix'][ind]])
                
            ### Plot the results
            # The weighted loss
            fig,ax = plt.subplots(figsize=(8,6))
            ax.scatter(complexity[:num_plot],test_losses_onloader_onemodel_parato, c='blue')
            ax.set_xlabel('Complexity (length)',fontsize=16)
            ax.set_ylabel('Weighted MSE (latent%d) for Pareto (blue)'%ind_var,fontsize=16)
            ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.axhline(1., color="grey", linestyle="--") # The threshold for the weighted error (likelihood)
            plt.xticks(list(np.sort(complexity)))
        
            # Create a second axis for the mse plots
            ax2=ax.twinx()
            ax2.scatter(complexity[:num_plot],mse_pred,color='r')
            ax2.set_yscale('log')
            ax2.set_ylabel('MSE for SR (red)',fontsize=16)
            
            print ('The weighted MSE for latent%d:'%ind_var,np.array(test_losses_onloader_onemodel_parato))
            plt.show()
            plt.close()
            #for i in range(len(complexity)):
            #    print ('raw expression:',data0['infix'][i])
            #    print ('*****')
            #    pprint (data0['infix'][i].replace('^','**'))
            #    print ('***********************************')
            
    
        # For the individual expressions
        
    print('Outputting good expressions of the first latent with final weighted mse less than 1 for individuals')
    with open('./data/sr/individual_good_model_latent%d_model_%s_operon_%s.csv'%(1, name,operon_name), 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['length','weighted MSE','expression'])
        
        maskindex = 6
        #interval=1 # 200 for fewer computational costs
        for ind_var in range(maskindex):
            test_name = operon_name+'_latent%d'%ind_var
            test_losses_onloader_onemodel_individual=[]
            
            # sr prediction: [len(expressions),500], i.e. [2000,500]
            sr_pred = torch.tensor(np.float32(np.load('./data/sr/pred_individuals_%s.npy'%test_name)))
            # expressions: [len(expressions),]
            data0=pd.read_csv('./data/sr/individuals_%s.csv'%test_name)
            # Complexity of the expressions
            complexity=pd.read_csv('./data/sr/individuals_%s.csv'%test_name)['length'].values[::interval]
            
            num_plot = 2000 #10 # number of individuals in the operon setting is 2000
            for ind in range(0,num_plot,interval):
                temp=test_epoch_replace(encoder, decoder, maskin, d_encode,device, test_loader, loss_fn,maskindex, ind, ind_var,min_temp,data_min,data_max,name,sr_pred) # Here we did not use the test_loader, but load the test data from scratch
                test_losses_onloader_onemodel_individual.append(temp)
                if ind_var==0 and temp<=1:
                #if True:
                    writer.writerow([complexity[ind],np.round(np.array(temp),3),data0['infix'][ind]])
                
            ### Plot the results
            fig,ax = plt.subplots(figsize=(8,6))
            ax.scatter(complexity,test_losses_onloader_onemodel_individual)
            ax.set_xlabel('Complexity (length)',fontsize=16)
            ax.set_ylabel('Weighted MSE (latent%d)'%ind_var,fontsize=16)
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.axhline(1., color="grey", linestyle="--") # The threshold for the weighted error (likelihood)
            #ax.legend(fontsize=11)
            plt.show()
            plt.close()
        
if __name__ == '__main__':
    main()