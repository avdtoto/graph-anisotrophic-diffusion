
from torch_geometric.utils import to_dense_adj, get_laplacian, dense_to_sparse
from torch_geometric.data import Data


import torch
import torch.nn as nn
import numpy as np
import scipy


def preprocessing_dataset(dataset, num_of_eigenvectors):

    # Calculate and store the vector field F

    data_dic = {}

    # Iterate through the first two elements of the dataset (for demonstration)
    for index in range(2):

        # Convert dataset element to dictionary
        data_dic[index] = dataset[index].to_dict()

        # Find the number of nodes for the graph
        num_nodes = dataset[index].num_nodes
        # Get a normalized value = sqrt((total probability=1)/number_of_nodes), which is assigned to all nodes
        norm_n = torch.FloatTensor(num_nodes, 1).fill_(1. / float(num_nodes)).sqrt()
        # Store the vector in the dictionary
        data_dic[index]["norm_n"] = norm_n
        # Create adjacency matrix to show edge connectivity, extracting it from the list by using [0] - so it's 16x16, not 16x16x1
        adj = to_dense_adj(dataset[index].edge_index)[0]
        # Sum up ones across columns - get a degree vector for each node
        node_deg_vec = adj.sum(axis=1, keepdim=True)
        # Create a matrix with diagonals filled with degrees for corresponding nodes
        node_deg_mat = torch.diag(node_deg_vec[:, 0])

        # Get the Laplacian matrix - subtracting D-A we get the degrees on the main diagonal, and -1 in other components for edges (0-1 = -1)
        lap_mat = get_laplacian(dataset[index].edge_index)
        L = to_dense_adj(edge_index = lap_mat[0], edge_attr = lap_mat[1])[0]

        # Calculate the eigenvalues and eigenvectors for the Laplacian matrix
        w = scipy.linalg.eigh(L.cpu(), b = node_deg_mat.cpu())
        eigenvalues, eigenvectors = w[0], w[1]
        # Optional: Print eigenvectors for a specified number of eigenvectors

        # Optionally, eigenvector values can be used (sorted in ascending order by default)
        k_eig_vec = torch.from_numpy(np.real(eigenvectors[:, :num_of_eigenvectors])).float()
        k_eig_val = torch.from_numpy(np.real(eigenvalues[:num_of_eigenvectors])).float()

        # If there are fewer nodes than the number of eigenvectors, pad with zeros
        if num_nodes < num_of_eigenvectors:
            zero_vec_1 = torch.zeros(num_nodes, num_of_eigenvectors - num_nodes)
            zero_vec_2 = torch.zeros(num_of_eigenvectors - num_nodes)

            # Clamp values to avoid division by zero
            zero_vec_1 = torch.clamp(zero_vec_1, min=1e-8)
            zero_vec_2 = torch.clamp(zero_vec_2, min=1e-8)

            k_eig_vec = torch.cat((k_eig_vec, zero_vec_1), dim = 1)
            k_eig_val = torch.cat((k_eig_val, zero_vec_2), dim = 0)

        # Store the k eigenvectors and eigenvalues in the dictionary
        data_dic[index]["k_eig_vec"] = k_eig_vec
        data_dic[index]["k_eig_val"] = k_eig_val

        # Calculate the Fiedler vector - for the three smallest eigenvalues, construct a vector field
        vec_idxs = [1, 2, 3] # List of eigenvector indices

        data_dic[index]['F_norm_edge'] = []
        data_dic[index]['F_dig'] = []

        for index_vec in vec_idxs:
            eig_vec_idx = eigenvectors[:, index_vec]
            # Construct a vector field based on adjacency matrix. If 1, there is a connection between nodes, replaced by the value
            # from the eigenvector, found as the difference between eigenvalues of node i and node j. If 0 or not connected, replace with a very small value to avoid nullification
            F = np.array(adj.cpu())

            for i in range(F.shape[0]):
                for j in range(F.shape[0]):

                    if F[i][j] == 1:
                        F[i][j] = eig_vec_idx[i] - eig_vec_idx[j]

                        # To maintain the same edge_index as the adj_matrix
                        if F[i][j] == 0:
                            F[i][j] = 1e-8

            # Normalize the vector field F

            F_norm = F

            for i in range(F.shape[0]):
                norm = np.linalg.norm(F[i], ord = 1)
                eps = 1e-20
                F_norm[i] = F_norm[i] / (norm + eps)


            F_dig = np.sum(F_norm, axis=0)

            F_norm = torch.from_numpy(F_norm)

            F_dig = torch.from_numpy(F_dig)

            F_norm_edge = dense_to_sparse(F_norm)[1]


            data_dic[index]['F_norm_edge'].append(F_norm_edge)
            data_dic[index]['F_dig'].append(F_dig)


    data_list = []

    for i in range(len(data_dic)):

        data_list.append(Data.from_dict(data_dic[i]))

    return data_list


# Calculate the average node degree in the training data

def average_node_degree(dataset):
  
  D = []
  
  for i in range(len(dataset)):

      adj = to_dense_adj(dataset[i].edge_index)[0]

      deg = adj.sum(axis=1, keepdim=True) # Degree of nodes, shape [N, 1]

      D.append(deg.squeeze())

  D = torch.cat(D, dim = 0)

  avg_d  = dict(lin=torch.mean(D),
          exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
          log=torch.mean(torch.log(D + 1)))


  return D, avg_d

