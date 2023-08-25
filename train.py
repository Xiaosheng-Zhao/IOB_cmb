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

### Define the loss function
#loss_fn = torch.nn.MSELoss()
def loss_fn(output, target, weight):
    loss = torch.mean((output - target)**2/weight)
    return loss


### Training function
def train_epoch(encoder, decoder, maskin, d_encode, device, dataloader, loss_fn, optimizer,epoch):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []

    # Iterate the dataloader
    for data_batch,data_batch_mean,param,err in dataloader:
        data_batch = data_batch.to(device)
        data_batch_mean = data_batch_mean.to(device)
        param = param.to(device)
        err = err.to(device)
        # Encode data
        encoded_data = encoder(param) 
        maskin_data = maskin(encoded_data)
        
        # Initialize the loss
        loss=torch.tensor(0,dtype=torch.float32).to(device)
        
        range_index=torch.arange(1, d_encode+1)
        for x_index in range(d_encode+1):
            mask1 = range_index.le(x_index).unsqueeze(0).to(device)
            loss+=loss_fn(data_batch_mean[:,0,:], decoder(maskin_data*mask1),err)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def valid_epoch(encoder, decoder, maskin, d_encode, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = [] # For the decoded (reconstructed data)
        conc_data = [] # Which is the real input data
        conc_data_mean = [] # For the real mean data (for each set of parameters)
        conc_param = [] # For the parameters
        conc_err = [] # For the cosmic variance
        for data_batch,data_batch_mean,param,err in dataloader:
            data_batch = data_batch.to(device)
            data_batch_mean = data_batch_mean.to(device)
            param = param.to(device)
            err = err.to(device)
            # Encode data
            encoded_data= encoder(param)
            maskin_data = maskin(encoded_data)
            # Decode data
            decoded_data = decoder(maskin_data)
            # Append the network output and the original data to the lists
            conc_out.append(decoded_data.cpu())
            conc_data.append(data_batch.cpu())
            conc_data_mean.append(data_batch_mean.cpu())
            conc_err.append(err.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_data = torch.cat(conc_data)
        conc_data_mean = torch.cat(conc_data_mean)
        conc_err = torch.cat(conc_err)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_data_mean[:,0,:],conc_err)
    return val_loss.data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="The basic name of encoder-decoder experiments",
                        type=str,
                        default='shallow96-test',
                        required=False)
    
    parser.add_argument("--epoch",
                        help="number of training epochs",
                        type=int,
                        default=2048,
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
    lr= 1e-5
    num_epochs =args.epoch #2048
    patience=20 # The patience for early stopping of the training. 
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
        
    ### Initialize the networks
    encoder = Encoder_cmb(encoded_space_dim=d_encode)
    decoder = Decoder_cmb(encoded_space_dim=d_encode)
    maskin = maskin_cmb()
    
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': maskin.parameters()},
        {'params': decoder.parameters()}
    ]
    
    ### Move both the encoder and the decoder to the selected device
    encoder.to(device)
    decoder.to(device)
    maskin.to(device)
    
    ### Define an optimizer (both for the encoder and the decoder)
    optim = torch.optim.Adam(params_to_optimize, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=5, factor=0.1)
    
    ### Initialize the earlystoping
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    ### The training part
    # Define the checkpoint path
    state_save_early_path='./model/checkpoint_%s.tar'%name
    
    number_of_params = sum(x.numel() for x in decoder.parameters())
    print(f"Number of parameters: {number_of_params}")
        
    for epoch in range(num_epochs):
        train_loss =train_epoch(encoder,decoder,maskin,d_encode,device,train_loader,loss_fn,optim,epoch)
        val_loss = valid_epoch(encoder,decoder,maskin,d_encode,device,valid_loader,loss_fn)
        scheduler.step(val_loss)
        print('\n EPOCH {}/{} \t train loss {} \t val loss {} \t name {}'.format(epoch + 1, num_epochs,train_loss,val_loss,name))
        model_state = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'maskin_state_dict': maskin.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'val_loss': val_loss
                }
        early_stopping(val_loss, model_state,state_save_early_path)
        if early_stopping.early_stop or epoch==num_epochs-1:
            if early_stopping.early_stop:
                print("Early stopping")
            print ("Stop training!")
            break
    

if __name__ == '__main__':
    main()