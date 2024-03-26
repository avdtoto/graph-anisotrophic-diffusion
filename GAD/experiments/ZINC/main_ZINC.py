import torch
import torch.optim as opt
import torch.nn as nn

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader
import os
import sys

from tqdm import tqdm

import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/")) 

from preprocessing import preprocessing_dataset, average_node_degree
from train_eval_ZINC import train_epoch, evaluate_network
from GAD_ZINC.gad import GAD

def train_ZINC(model, optimizer, train_loader, val_loader, device, num_epochs, min_lr):

    loss_fn = nn.L1Loss()

    scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, 
                                                     patience=15,
                                                   threshold=0.004,
                                                   verbose=True)

    epoch_train_MAEs, epoch_val_MAEs = [], []
    
    Best_val_mae = 10

    print("Start training")

    for epoch in range(num_epochs):
        
        if optimizer.param_groups[0]['lr'] < min_lr:
            print("lr equal to min_lr: exist")
            break
        
        epoch_train_mae, optimizer = train_epoch(model ,train_loader, optimizer, device, loss_fn)
        epoch_val_mae = evaluate_network(model,  val_loader, device)

        epoch_train_MAEs.append(epoch_train_mae)
        epoch_val_MAEs.append(epoch_val_mae)

        scheduler.step(epoch_val_mae)
        if(epoch_val_mae < Best_val_mae):
            Best_val_mae =  epoch_val_mae
            torch.save(model, 'model.pth')

        torch.save(model, 'model_running.pth')

        print("")
        print("epoch_idx", epoch)
        print("epoch_train_MAE", epoch_train_mae)
        print("epoch_val_MAE", epoch_val_mae)
        
    print("Finish training")

def main():

    batch_size=16
    k=30

    hid_dim=65
    use_graph_norm=True
    use_batch_norm=True
    dropout=0
    readout='mean'
    aggregators='mean dir_der max min'
    scalers='identity amplification attenuation'
    use_edge_fts=True
    towers=5
    type_net='tower'
    use_residual=True
    use_diffusion=True
    diffusion_method='implicit'
    n_layers=4


    lr=1e-3
    weight_decay=3e-6
    num_epochs=300
    min_lr=1e-5

    print("downloading the dataset (ZINC)")
    dataset_train = ZINC(root='/tmp/a', subset=True)
    dataset_val = ZINC(root='/tmp/a', subset=True, split='val')
    dataset_test = ZINC(root='/tmp/a', subset=True, split='test')

    print("dataset_train contains ", len(dataset_train), "samples")
    print("dataset_val contains ", len(dataset_val), "samples")
    print("dataset_test contains ", len(dataset_test), "samples")

    print("data preprocessing: calculate and store the vector field F, etc.")

    D, avg_d = average_node_degree(dataset_train)
    dataset_train = preprocessing_dataset(dataset_train, k)
    dataset_val = preprocessing_dataset(dataset_val, k)
    dataset_test = preprocessing_dataset(dataset_test, k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset = dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset =  dataset_test, batch_size=batch_size, shuffle=False)

    print("create GAD model")

    model = GAD(num_atom_type = 28, num_bond_type = 4, hid_dim = hid_dim, graph_norm = use_graph_norm,
               batch_norm = use_batch_norm, dropout = dropout, readout = readout, aggregators = aggregators,
               scalers = scalers, edge_fts = use_edge_fts, avg_d = avg_d, D = D, device = device, towers= towers,
               type_net = type_net, residual = use_residual, use_diffusion = use_diffusion,
               diffusion_method = diffusion_method, k = k, n_layers = n_layers)


    model = model.to(device)

    optimizer = opt.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ZINC(model, optimizer, train_loader, val_loader, device, num_epochs = num_epochs, min_lr = min_lr)

    print("Uploading the best model")

    model_ = torch.load('/tmp/a/GAN_train/model.pth')

    test_mae = evaluate_network(model_, test_loader, device)
    val_mae = evaluate_network(model_, val_loader, device)
    train_mae = evaluate_network(model_, train_loader, device)

    print("")
    print("Best Train MAE: {:.4f}".format(train_mae))
    print("Best Val MAE: {:.4f}".format(val_mae))
    print("Best Test MAE: {:.4f}".format(test_mae))


main()
