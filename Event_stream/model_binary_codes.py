### The base code from https://github.com/uzh-rpg/aegnn has been modified to incorporate Diffusion Loss.

#Need to change batch size variable as per the batch size chosen in the training file.
import torch
import torch_geometric

from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.transforms import Cartesian

from aegnn.models.layer import MaxPooling, MaxPoolingX
import os
import ot
import scipy
import numpy as np

class GraphRes(torch.nn.Module):

    def __init__(self, dataset, input_shape: torch.Tensor, num_outputs: int, pooling_size=(16, 12),
                 bias: bool = False, root_weight: bool = False):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        # print("input_shape",input_shape)
        # print("dim",dim)
        # Set dataset specific hyper-parameters.
        if dataset == "ncars":
            kernel_size = 2
            n = [1, 8, 16, 16, 16, 32, 32, 32, 32]
            pooling_outputs = 32
        elif dataset == "ncaltech101" or dataset == "gen1":
            kernel_size = 8
            n = [1, 16, 32, 32, 32, 128, 128, 128]
            pooling_outputs = 128
        else:
            raise NotImplementedError(f"No model parameters for dataset {dataset}")

        self.num_layers = 1

        self.conv1 = SplineConv(n[0], n[1], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=n[1])
        self.conv2 = SplineConv(n[1], n[2], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=n[2])

        self.conv3 = SplineConv(n[2], n[3], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=n[3])
        self.conv4 = SplineConv(n[3], n[4], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm4 = BatchNorm(in_channels=n[4])

        self.conv5 = SplineConv(n[4], n[5], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm5 = BatchNorm(in_channels=n[5])
        self.pool5 = MaxPooling(pooling_size, transform=Cartesian(norm=True, cat=False))

        self.conv6 = SplineConv(n[5], n[6], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm6 = BatchNorm(in_channels=n[6])
        self.conv7 = SplineConv(n[6], n[7], dim=dim, kernel_size=kernel_size, bias=bias, root_weight=root_weight)
        self.norm7 = BatchNorm(in_channels=n[7])

        self.pool7 = MaxPoolingX(input_shape[:2] // 4, size=16)
        self.fc = Linear(pooling_outputs * 16, out_features=num_outputs, bias=bias)


        self.use_tmd = True
        if self.use_tmd:
            L_latent = 1
            self.pi_list = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(L_latent, 128),
                                                        torch.nn.ReLU(),
                                                        torch.nn.Linear(128, 1),
                                                        torch.nn.Sigmoid()) for _ in range(self.num_layers)])
            self.proj_list = torch.nn.ModuleList([torch.nn.Linear(1, 4000)] +
                                            [torch.nn.Linear(1, 4000) for _ in range(self.num_layers-1)])
            self.dt_0 = torch.nn.Parameter(torch.cuda.FloatTensor([0.1]))
            self.dt_1 = torch.nn.Parameter(torch.cuda.FloatTensor([0.1]))
            self.dt_2 = torch.nn.Parameter(torch.cuda.FloatTensor([0.1]))
            self.dt_3 = torch.nn.Parameter(torch.cuda.FloatTensor([0.1]))

    def TMD_map(self, x, idx,adj_matrix):
        # input x if of size [B, d, N]

        batch_size = 4

        tmd_sample = False

        L = torch.zeros(batch_size,batch_size).cuda()

        if x.shape[0] % batch_size == 0:
            if adj_matrix.shape[1] == x.shape[0]:

                tmd_sample = True
                #random features
                unique_matrix = torch.randn_like(x)
                
                #Set no. of iterations of C-WL
                iterations = 4

                random_x = x.clone()
                
                K_epsilon_list = []

                for i in range(iterations):
                    #C-WL
                    random_x = torch.transpose(random_x, 0, 1) + torch.matmul(adj_matrix, random_x).sum(dim=(2))

                    var,mean = torch.var_mean(random_x, unbiased=False)

                    normalized_random_x = torch.div((random_x - mean),torch.sqrt(var))
                    
                    #Binary Codes Implemenatation
                    rff_w = torch.normal(0, 1, size=(normalized_random_x.shape[0]//batch_size,1))
                    rff_b = torch.normal(0, 2.0 * np.pi, size=(batch_size, 1))

                    Qt_x = torch.cos(torch.matmul(normalized_random_x.reshape(batch_size,normalized_random_x.shape[0]//batch_size).cuda(), rff_w.cuda())+ rff_b.cuda() ).cuda()

                    t = torch.FloatTensor(batch_size, 1).uniform_(-1, 1).cuda()

                    sgn = (Qt_x + t).cpu()

                    sgn.apply_(lambda x: -1 if x < 0 else 1)

                    sgn.apply_(lambda x: (1 + x)/2)

                    K_epsilon_iter = torch.cdist(sgn, sgn, p=0).cuda()

                    K_epsilon_list.append(K_epsilon_iter)


                #Find avg. K_epsilon for all iterations.
                avg_K_epsilon = torch.zeros_like(K_epsilon_list[0])
                
                for each_element in K_epsilon_list:
                    avg_K_epsilon = avg_K_epsilon + each_element
                
                #TMD Layer implementation
                K_epsilon = torch.div(avg_K_epsilon,iterations)

                epsilon = 0.25

                q_epsilon_tilde = (K_epsilon).sum(dim=1)

                D_epsilon_tilde = torch.diag_embed(0.25 / q_epsilon_tilde)

                K_tilde  = torch.matmul(K_epsilon, D_epsilon_tilde)
               
                D_tilde = torch.diag_embed(K_tilde.sum(dim=1))

                # Diffusion operator
                L =  1 / epsilon * (torch.inverse(D_tilde).matmul(K_tilde)) - torch.eye(K_tilde.shape[1]).to(x.device)

        return L, tmd_sample


    def forward(self, data: torch_geometric.data.Batch) -> torch.Tensor:


        adj_matrix = torch_geometric.utils.to_dense_adj(data.edge_index)

        l = 0

        L, tmd_check = self.TMD_map(data.x, l,adj_matrix)  
     
        
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)


        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)

        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc


        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))

        data.x = self.norm5(data.x)

        #Correct features using Diffusion Operator
        if self.use_tmd:
            if tmd_check :
                if l == 0:
                    dt = self.dt_0
                elif l ==1:
                    dt = self.dt_1
                elif l == 2:
                    dt = self.dt_2
                
                batch_size = 4

                data.x = (data.x) + dt * torch.matmul(L, data.x.reshape(batch_size,int(data.x.shape[0] / batch_size * data.x.shape[1]))).reshape(data.x.shape[0],data.x.shape[1])


        data = self.pool5(data.x, pos=data.pos, batch=data.batch, edge_index=data.edge_index, return_data_obj=True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))

        data.x = self.norm6(data.x)


        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
       
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc
        
        x = self.pool7(data.x, pos=data.pos[:, :2], batch=data.batch)
        x = x.view(-1, self.fc.in_features)

        return self.fc(x)
