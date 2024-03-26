
from torch_geometric.utils import to_dense_adj, get_laplacian, dense_to_sparse
from torch_geometric.data import Data


import torch
import torch.nn as nn
import numpy as np
import scipy

# def preprocessing_dataset(dataset, num_of_eigenvectors):
#
#     # Calculate and store the vector field F
#
#     data_dic = {}
#
#     for index in range(len(dataset)):
#
#
#         data_dic[index] = dataset[index].to_dict()
#
#         num_nodes = dataset[index].num_nodes
#
#         norm_n = torch.FloatTensor(num_nodes, 1).fill_(1. / float(num_nodes)).sqrt()
#
#         data_dic[index]["norm_n"] = norm_n
#
#         adj = to_dense_adj(dataset[index].edge_index)[0]
#
#         node_deg_vec = adj.sum(axis=1, keepdim=True)
#
#
#         node_deg_mat = torch.diag(node_deg_vec[:, 0])
#
#
#         lap_mat = get_laplacian(dataset[index].edge_index)
#         L = to_dense_adj(edge_index = lap_mat[0], edge_attr = lap_mat[1])[0]
#
#         # calculate the  eigenvalues and the eigenvectors for the lap_matrix
#
#         w = scipy.linalg.eigh(L.cpu(), b = node_deg_mat.cpu())
#         eigenvalues, eigenvectors = w[0], w[1]
#
#
#         # sort the eigenvectors in ascending order
#
#         lowest_eig_vec_idx = eigenvalues.argsort()[1]
#
#         # take the eigenvector corresponding to the lowest eigenvalue
#
#         lowest_eig_vec = torch.from_numpy(np.real(eigenvectors[:, lowest_eig_vec_idx])).float()
#
#
#
#         # calculate the k lowers eigenvalues and the corresponding eigenvectors
#
#         eigenvectors = eigenvectors[:, eigenvalues.argsort()]  # increasing order
#         k_eig_vec = torch.from_numpy(np.real(eigenvectors[:, :num_of_eigenvectors])).float()
#         eigenvalues.sort()
#         k_eig_val = torch.from_numpy(np.real(eigenvalues[:num_of_eigenvectors])).float()
#         if num_nodes < num_of_eigenvectors:
#             # padding by zeros
#             zero_vec_1 = torch.zeros(num_nodes, num_of_eigenvectors - num_nodes)
#             zero_vec_2 = torch.zeros(num_of_eigenvectors - num_nodes)
#
#             zero_vec_1 = torch.clamp(zero_vec_1, min=1e-8)
#             zero_vec_2 = torch.clamp(zero_vec_2, min=1e-8)
#
#             k_eig_vec = torch.cat((k_eig_vec, zero_vec_1), dim = 1)
#             k_eig_val = torch.cat((k_eig_val, zero_vec_2), dim = 0)
#
#
#         data_dic[index]["k_eig_vec"] = k_eig_vec
#         data_dic[index]["k_eig_val"] = k_eig_val
#
#
#
#
#         F = np.array(adj.cpu())
#
#
#         for i in range(F.shape[0]):
#             for j in range(F.shape[0]):
#
#                 if F[i][j]==1:
#                     F[i][j] = lowest_eig_vec[i] - lowest_eig_vec[j]
#
#                     # just to have the same edge_index as the adj_matrix
#                     if F[i][j] == 0:
#                         F[i][j] = 1e-8
#
#         F_norm = F
#         # normalize the vector field F
#
#         for i in range(F.shape[0]):
#             norm = np.linalg.norm(F[i], ord = 1)
#             eps = 1e-20
#             F_norm[i] = F_norm[i] / (norm + eps)
#
#
#         F_dig = np.sum(F_norm, axis=0)
#
#         F_norm = torch.from_numpy(F_norm)
#
#         F_dig = torch.from_numpy(F_dig)
#
#         F_norm_edge = dense_to_sparse(F_norm)[1]
#
#         data_dic[index]["F_norm_edge"] = F_norm_edge
#
#         data_dic[index]["F_dig"] = F_dig
#
#
#     data_list = []
#
#     for i in range(len(data_dic)):
#
#         data_list.append(Data.from_dict(data_dic[i]))
#
#     return data_list


from torch_geometric.utils import to_dense_adj, get_laplacian, dense_to_sparse
from torch_geometric.data import Data

import torch
import torch.nn as nn
import numpy as np
import scipy

def preprocessing_dataset(dataset, num_of_eigenvectors):

    # Calculate and store the vector field F

    data_dic = {}

    # for index in range(len(dataset)):
    for index in range(2):


        data_dic[index] = dataset[index].to_dict()


        # находим число нод для графа
        num_nodes = dataset[index].num_nodes
        # получаем нормированное значение = sqrt((полная вероятность=1)/число_нод), которое записываем для всех вершин
        norm_n = torch.FloatTensor(num_nodes, 1).fill_(1. / float(num_nodes)).sqrt()
        # записываем вектор в словарь
        data_dic[index]["norm_n"] = norm_n
        # создаем adjacency matrix для отображения связности ребер и вынимаем его из списка за счет [0] - не 16,16,1 а просто 16,16
        adj = to_dense_adj(dataset[index].edge_index)[0]
        # суммируем единицы по столбцам - получаем вектор степеней каждой ноды
        node_deg_vec = adj.sum(axis=1, keepdim=True)
        # создаем матрицу с диагональю заполненной степенями для соответствующей ноды
        node_deg_mat = torch.diag(node_deg_vec[:, 0])


        lap_mat = get_laplacian(dataset[index].edge_index)
        # получили матрицу Лапласа - вычли D-A по главной диагонали получили степени вершин, а в остальных компонентах связности получили -1 (0-1 = -1)
        L = to_dense_adj(edge_index = lap_mat[0], edge_attr = lap_mat[1])[0]

        # calculate the  eigenvalues and the eigenvectors for the lap_matrix
        w = scipy.linalg.eigh(L.cpu(), b = node_deg_mat.cpu())
        eigenvalues, eigenvectors = w[0], w[1]
        # print(eigenvectors[:, :num_of_eigenvectors])


        ##########################

        # проверим гипотезу о собственных векторах - гипотеза верна, нет необходимости доставать собственные числа и делать сортировку.
        #  Можно использовать значение eigenvectors(default - sort asc)

        k_eig_vec = torch.from_numpy(np.real(eigenvectors[:, :num_of_eigenvectors])).float()
        k_eig_val = torch.from_numpy(np.real(eigenvalues[:num_of_eigenvectors])).float()

        if num_nodes < num_of_eigenvectors:
            # padding by zeros
            zero_vec_1 = torch.zeros(num_nodes, num_of_eigenvectors - num_nodes)
            zero_vec_2 = torch.zeros(num_of_eigenvectors - num_nodes)

            zero_vec_1 = torch.clamp(zero_vec_1, min=1e-8)
            zero_vec_2 = torch.clamp(zero_vec_2, min=1e-8)

            k_eig_vec = torch.cat((k_eig_vec, zero_vec_1), dim = 1)
            k_eig_val = torch.cat((k_eig_val, zero_vec_2), dim = 0)


        data_dic[index]["k_eig_vec"] = k_eig_vec
        data_dic[index]["k_eig_val"] = k_eig_val

        # Посчитаем вектор Фидлера - индекс - (1)
        # для трех наименьших собственных векторов построим векторное поле
        vec_idxs = [1, 2, 3] # список индексов собственных векторов

        data_dic[index]['F_norm_edge'] = []
        data_dic[index]['F_dig'] = []

        for index_vec in vec_idxs:
            eig_vec_idx = eigenvectors[:, index_vec]
            # составляем векторное поле на основе матрицы смежности ребер. Если 1 - еcть связь между вершинами и мы ее заменяем на значение
            # из собственного вектора, которую находим как разниWу между собственным числом i вершины и j вершины, если значение равно 1 и 0 - если диагональ и не связаны, но заменяем на очень малое, чтобы не обращалось все в ноль
            F = np.array(adj.cpu())

            for i in range(F.shape[0]):
                for j in range(F.shape[0]):

                    if F[i][j]==1:
                        F[i][j] = eig_vec_idx[i] - eig_vec_idx[j]

                        # just to have the same edge_index as the adj_matrix
                        if F[i][j] == 0:
                            F[i][j] = 1e-8

            F_norm = F

            # normalize the vector field F
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

