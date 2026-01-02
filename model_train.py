#!/usr/bin/env python
"""Convolutional Neural Network Training Functions

Functions for building and training a (UNET) Convolutional Neural Network on
images of the Moon and binary ring targets.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import h5py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import utils.template_match_target as tmt
import utils.processing as proc


########################
def get_param_i(param, i):
    """Gets correct parameter for iteration i.

    Parameters
    ----------
    param : list
        List of model hyperparameters to be iterated over.
    i : integer
        Hyperparameter iteration.

    Returns
    -------
    Correct hyperparameter for iteration i.
    """
    if len(param) > i:
        return param[i]
    else:
        return param[0]

########################
class CraterDataset(Dataset):
    """Custom Dataset for crater images with data augmentation.
    
    Parameters
    ----------
    data : numpy.ndarray
        Input images with shape (N, H, W, C).
    target : numpy.ndarray
        Target masks with shape (N, H, W).
    augment : bool
        Whether to apply data augmentation.
    """
    def __init__(self, data, target, augment=True):
        # Convert from (N, H, W, C) to (N, C, H, W) for PyTorch
        self.data = torch.from_numpy(data).permute(0, 3, 1, 2).float()
        self.target = torch.from_numpy(target).float()
        self.augment = augment
        self.L, self.W = data.shape[1], data.shape[2]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        d = self.data[idx].clone()
        t = self.target[idx].clone()
        
        if self.augment:
            # Horizontal/vertical flips
            if np.random.rand() > 0.5:
                d = torch.flip(d, [2])  # flip width
                t = torch.flip(t, [1])
            if np.random.rand() > 0.5:
                d = torch.flip(d, [1])  # flip height
                t = torch.flip(t, [0])
            
            # Random shifts and rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1)
            v = np.random.randint(-npix, npix + 1)
            r = np.random.randint(0, 4)  # 90 degree rotations
            
            # Pad for shifts
            d = F.pad(d, (npix, npix, npix, npix), mode='constant', value=0)
            t = F.pad(t, (npix, npix, npix, npix), mode='constant', value=0)
            
            # Apply shifts
            d = d[:, npix + v:self.L + v + npix, npix + h:self.W + h + npix]
            t = t[npix + v:self.L + v + npix, npix + h:self.W + h + npix]
            
            # Apply rotation
            if r > 0:
                d = torch.rot90(d, r, [1, 2])
                t = torch.rot90(t, r, [0, 1])
        
        return d, t

########################
def custom_image_generator(data, target, batch_size=32):
    """Custom image generator that manipulates image/target pairs to prevent
    overfitting in the Convolutional Neural Network.

    Parameters
    ----------
    data : array
        Input images.
    target : array
        Target images.
    batch_size : int, optional
        Batch size for image manipulation.

    Yields
    ------
    Manipulated images and targets.
        
    """
    L, W = data[0].shape[0], data[0].shape[1]
    while True:
        for i in range(0, len(data), batch_size):
            d, t = data[i:i + batch_size].copy(), target[i:i + batch_size].copy()

            # Random color inversion
            # for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
            #     d[j][d[j] > 0.] = 1. - d[j][d[j] > 0.]

            # Horizontal/vertical flips
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.fliplr(d[j]), np.fliplr(t[j])      # left/right
            for j in np.where(np.random.randint(0, 2, batch_size) == 1)[0]:
                d[j], t[j] = np.flipud(d[j]), np.flipud(t[j])      # up/down

            # Random up/down & left/right pixel shifts, 90 degree rotations
            npix = 15
            h = np.random.randint(-npix, npix + 1, batch_size)    # Horizontal shift
            v = np.random.randint(-npix, npix + 1, batch_size)    # Vertical shift
            r = np.random.randint(0, 4, batch_size)               # 90 degree rotations
            for j in range(batch_size):
                d[j] = np.pad(d[j], ((npix, npix), (npix, npix), (0, 0)),
                              mode='constant')[npix + h[j]:L + h[j] + npix,
                                               npix + v[j]:W + v[j] + npix, :]
                t[j] = np.pad(t[j], (npix,), mode='constant')[npix + h[j]:L + h[j] + npix, 
                                                              npix + v[j]:W + v[j] + npix]
                d[j], t[j] = np.rot90(d[j], r[j]), np.rot90(t[j], r[j])
            yield (d, t)

########################
def get_metrics(data, craters, dim, model, device, beta=1):
    """Function that prints pertinent metrics at the end of each epoch. 

    Parameters
    ----------
    data : tuple
        Input images and target masks.
    craters : hdf5
        Pandas arrays of human-counted crater data. 
    dim : int
        Dimension of input images (assumes square).
    model : torch.nn.Module
        PyTorch model
    device : torch.device
        Device to run inference on.
    beta : int, optional
        Beta value when calculating F-beta score. Defaults to 1.
    """
    X, Y = data[0], data[1]

    # Get csvs of human-counted craters
    csvs = []
    minrad, maxrad, cutrad, n_csvs = 3, 50, 0.8, len(X)
    diam = 'Diameter (pix)'
    for i in range(n_csvs):
        csv = craters[proc.get_id(i)]
        # remove small/large/half craters
        csv = csv[(csv[diam] < 2 * maxrad) & (csv[diam] > 2 * minrad)]
        csv = csv[(csv['x'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['y'] + cutrad * csv[diam] / 2 <= dim)]
        csv = csv[(csv['x'] - cutrad * csv[diam] / 2 > 0)]
        csv = csv[(csv['y'] - cutrad * csv[diam] / 2 > 0)]
        if len(csv) < 3:    # Exclude csvs with few craters
            csvs.append([-1])
        else:
            csv_coords = np.asarray((csv['x'], csv['y'], csv[diam] / 2)).T
            csvs.append(csv_coords)

    # Calculate custom metrics
    print("")
    print("*********Custom Loss*********")
    recall, precision, fscore = [], [], []
    frac_new, frac_new2, maxrad = [], [], []
    err_lo, err_la, err_r = [], [], []
    frac_duplicates = []
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).permute(0, 3, 1, 2).float().to(device)
        preds = model(X_tensor).cpu().numpy()
    
    for i in range(n_csvs):
        if len(csvs[i]) < 3:
            continue
        (N_match, N_csv, N_detect, maxr,
         elo, ela, er, frac_dupes) = tmt.template_match_t2c(preds[i], csvs[i],
                                                            rmv_oor_csvs=0)
        if N_match > 0:
            p = float(N_match) / float(N_match + (N_detect - N_match))
            r = float(N_match) / float(N_csv)
            f = (1 + beta**2) * (r * p) / (p * beta**2 + r)
            diff = float(N_detect - N_match)
            fn = diff / (float(N_detect) + diff)
            fn2 = diff / (float(N_csv) + diff)
            recall.append(r)
            precision.append(p)
            fscore.append(f)
            frac_new.append(fn)
            frac_new2.append(fn2)
            maxrad.append(maxr)
            err_lo.append(elo)
            err_la.append(ela)
            err_r.append(er)
            frac_duplicates.append(frac_dupes)
        else:
            print("skipping iteration %d,N_csv=%d,N_detect=%d,N_match=%d" %
                  (i, N_csv, N_detect, N_match))

    # Calculate binary cross-entropy loss
    Y_tensor = torch.from_numpy(Y).float().to(device)
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(torch.from_numpy(preds).float().to(device), Y_tensor).item()
    
    print("binary XE score = %f" % loss)
    if len(recall) > 3:
        print("mean and std of N_match/N_csv (recall) = %f, %f" %
              (np.mean(recall), np.std(recall)))
        print("""mean and std of N_match/(N_match + (N_detect-N_match))
              (precision) = %f, %f""" % (np.mean(precision), np.std(precision)))
        print("mean and std of F_%d score = %f, %f" %
              (beta, np.mean(fscore), np.std(fscore)))
        print("""mean and std of (N_detect - N_match)/N_detect (fraction
              of craters that are new) = %f, %f""" %
              (np.mean(frac_new), np.std(frac_new)))
        print("""mean and std of (N_detect - N_match)/N_csv (fraction of
              "craters that are new, 2) = %f, %f""" %
              (np.mean(frac_new2), np.std(frac_new2)))
        print("median and IQR fractional longitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_lo), np.percentile(err_lo, 25),
               np.percentile(err_lo, 75)))
        print("median and IQR fractional latitude diff = %f, 25:%f, 75:%f" %
              (np.median(err_la), np.percentile(err_la, 25),
               np.percentile(err_la, 75)))
        print("median and IQR fractional radius diff = %f, 25:%f, 75:%f" %
              (np.median(err_r), np.percentile(err_r, 25),
               np.percentile(err_r, 75)))
        print("mean and std of frac_duplicates: %f, %f" %
              (np.mean(frac_duplicates), np.std(frac_duplicates)))
        print("""mean and std of maximum detected pixel radius in an image =
              %f, %f""" % (np.mean(maxrad), np.std(maxrad)))
        print("""absolute maximum detected pixel radius over all images =
              %f""" % np.max(maxrad))
        print("")

########################
class UNet(nn.Module):
    """U-Net architecture for crater detection.
    
    Parameters
    ----------
    n_filters : int
        Number of filters in the first layer.
    FL : int
        Filter/kernel size.
    init : str
        Weight initialization method.
    lmbda : float
        L2 regularization parameter.
    drop : float
        Dropout rate.
    """
    def __init__(self, n_filters=112, FL=3, init='he_normal', lmbda=1e-6, drop=0.15):
        super(UNet, self).__init__()
        self.lmbda = lmbda
        
        # Encoder
        self.conv1_1 = nn.Conv2d(1, n_filters, FL, padding=FL//2)
        self.conv1_2 = nn.Conv2d(n_filters, n_filters, FL, padding=FL//2)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2_1 = nn.Conv2d(n_filters, n_filters * 2, FL, padding=FL//2)
        self.conv2_2 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL//2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3_1 = nn.Conv2d(n_filters * 2, n_filters * 4, FL, padding=FL//2)
        self.conv3_2 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL//2)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.conv4_1 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL//2)
        self.conv4_2 = nn.Conv2d(n_filters * 4, n_filters * 4, FL, padding=FL//2)
        
        # Decoder
        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop5 = nn.Dropout2d(drop)
        self.conv5_1 = nn.Conv2d(n_filters * 8, n_filters * 2, FL, padding=FL//2)
        self.conv5_2 = nn.Conv2d(n_filters * 2, n_filters * 2, FL, padding=FL//2)
        
        self.up6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop6 = nn.Dropout2d(drop)
        self.conv6_1 = nn.Conv2d(n_filters * 4, n_filters, FL, padding=FL//2)
        self.conv6_2 = nn.Conv2d(n_filters, n_filters, FL, padding=FL//2)
        
        self.up7 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop7 = nn.Dropout2d(drop)
        self.conv7_1 = nn.Conv2d(n_filters * 2, n_filters, FL, padding=FL//2)
        self.conv7_2 = nn.Conv2d(n_filters, n_filters, FL, padding=FL//2)
        
        # Final output
        self.conv_final = nn.Conv2d(n_filters, 1, 1)
        
        # Initialize weights
        self._initialize_weights(init)
    
    def _initialize_weights(self, init):
        """Initialize weights based on the initialization method."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if init == 'he_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif init == 'glorot_uniform':
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        a1 = F.relu(self.conv1_1(x))
        a1 = F.relu(self.conv1_2(a1))
        a1P = self.pool1(a1)
        
        a2 = F.relu(self.conv2_1(a1P))
        a2 = F.relu(self.conv2_2(a2))
        a2P = self.pool2(a2)
        
        a3 = F.relu(self.conv3_1(a2P))
        a3 = F.relu(self.conv3_2(a3))
        a3P = self.pool3(a3)
        
        # Bottleneck
        u = F.relu(self.conv4_1(a3P))
        u = F.relu(self.conv4_2(u))
        
        # Decoder
        u = self.up5(u)
        u = torch.cat([a3, u], dim=1)
        u = self.drop5(u)
        u = F.relu(self.conv5_1(u))
        u = F.relu(self.conv5_2(u))
        
        u = self.up6(u)
        u = torch.cat([a2, u], dim=1)
        u = self.drop6(u)
        u = F.relu(self.conv6_1(u))
        u = F.relu(self.conv6_2(u))
        
        u = self.up7(u)
        u = torch.cat([a1, u], dim=1)
        u = self.drop7(u)
        u = F.relu(self.conv7_1(u))
        u = F.relu(self.conv7_2(u))
        
        # Final output
        u = self.conv_final(u)
        u = torch.sigmoid(u)
        u = u.squeeze(1)  # Remove channel dimension for output
        
        return u
    
    def get_l2_loss(self):
        """Calculate L2 regularization loss for all conv layers."""
        l2_loss = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                l2_loss += torch.sum(m.weight ** 2)
        return self.lmbda * l2_loss

########################
def build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters):
    """Function that builds the (UNET) convolutional neural network. 

    Parameters
    ----------
    dim : int
        Dimension of input images (assumes square).
    learn_rate : float
        Learning rate.
    lmbda : float
        Convolution2D regularization parameter. 
    drop : float
        Dropout fraction.
    FL : int
        Filter length.
    init : string
        Weight initialization type.
    n_filters : int
        Number of filters in each layer.

    Returns
    -------
    model : torch.nn.Module
        Constructed PyTorch model.
    """
    print('Making UNET model...')
    model = UNet(n_filters=n_filters, FL=FL, init=init, lmbda=lmbda, drop=drop)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    return model

########################
def train_and_test_model(Data, Craters, MP, i_MP):
    """Function that trains, tests and saves the model, printing out metrics
    after each model. 

    Parameters
    ----------
    Data : dict
        Inputs and Target Moon data.
    Craters : dict
        Human-counted crater data.
    MP : dict
        Contains all relevant parameters.
    i_MP : int
        Iteration number (when iterating over hypers).
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Static params
    dim, nb_epoch, bs = MP['dim'], MP['epochs'], MP['bs']

    # Iterating params
    FL = get_param_i(MP['filter_length'], i_MP)
    learn_rate = get_param_i(MP['lr'], i_MP)
    n_filters = get_param_i(MP['n_filters'], i_MP)
    init = get_param_i(MP['init'], i_MP)
    lmbda = get_param_i(MP['lambda'], i_MP)
    drop = get_param_i(MP['dropout'], i_MP)

    # Build model
    model = build_model(dim, learn_rate, lmbda, drop, FL, init, n_filters)
    model = model.to(device)
    
    # Create datasets and dataloaders
    train_dataset = CraterDataset(Data['train'][0], Data['train'][1], augment=True)
    dev_dataset = CraterDataset(Data['dev'][0], Data['dev'][1], augment=True)
    
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=True, num_workers=0)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    criterion = nn.BCELoss()
    
    # Training loop
    n_samples = MP['n_train']
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(nb_epoch):
        print(f"\nEpoch {epoch + 1}/{nb_epoch}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L2 regularization
            loss = loss + model.get_l2_loss()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / train_batches
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dev_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Get metrics on dev set
        get_metrics(Data['dev'], Craters['dev'], dim, model, device)

    # Save model
    if MP['save_models'] == 1:
        # Create directory if it doesn't exist
        save_dir = os.path.dirname(MP['save_dir'])
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_val_loss,
        }, MP['save_dir'])
        print(f"Model saved to {MP['save_dir']}")

    print("###################################")
    print("##########END_OF_RUN_INFO##########")
    print("""learning_rate=%e, batch_size=%d, filter_length=%e, n_epoch=%d
          n_train=%d, img_dimensions=%d, init=%s, n_filters=%d, lambda=%e
          dropout=%f""" % (learn_rate, bs, FL, nb_epoch, MP['n_train'],
                           MP['dim'], init, n_filters, lmbda, drop))
    get_metrics(Data['test'], Craters['test'], dim, model, device)
    print("###################################")
    print("###################################")

########################
def get_models(MP):
    """Top-level function that loads data files and calls train_and_test_model.

    Parameters
    ----------
    MP : dict
        Model Parameters.
    """
    dir = MP['dir']
    n_train, n_dev, n_test = MP['n_train'], MP['n_dev'], MP['n_test']

    # Load data
    train = h5py.File('%strain_images.hdf5' % dir, 'r')
    dev = h5py.File('%sdev_images.hdf5' % dir, 'r')
    test = h5py.File('%stest_images.hdf5' % dir, 'r')
    Data = {
        'train': [train['input_images'][:n_train].astype('float32'),
                  train['target_masks'][:n_train].astype('float32')],
        'dev': [dev['input_images'][:n_dev].astype('float32'),
                  dev['target_masks'][:n_dev].astype('float32')],
        'test': [test['input_images'][:n_test].astype('float32'),
                 test['target_masks'][:n_test].astype('float32')]
    }
    train.close()
    dev.close()
    test.close()

    # Rescale, normalize, add extra dim
    proc.preprocess(Data)

    # Load ground-truth craters
    Craters = {
        'train': pd.HDFStore('%strain_craters.hdf5' % dir, 'r'),
        'dev': pd.HDFStore('%sdev_craters.hdf5' % dir, 'r'),
        'test': pd.HDFStore('%stest_craters.hdf5' % dir, 'r')
    }

    # Iterate over parameters
    for i in range(MP['N_runs']):
        train_and_test_model(Data, Craters, MP, i)
