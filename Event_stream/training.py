### The base code from https://github.com/uzh-rpg/aegnn has been modified to incorporate Diffusion Loss.

import aegnn
import argparse
import itertools
import logging
import os
import pandas as pd
import torch
import torch_geometric

from torch_geometric.data import Data
from tqdm import tqdm
from typing import List

from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from typing import Iterable, Tuple
torch.cuda.empty_cache()
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", default=3.0, help="radius of radius graph generation")
    parser.add_argument("--max-num-neighbors", default=32, help="max. number of neighbors in graph")
    parser.add_argument("--pooling-size", default=(10, 10))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def sample_batch(batch_idx: torch.Tensor, num_samples: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Samples a subset of graphs in a batch and returns the sampled nodes and batch_idx.

    >> batch = torch.from_numpy(np.random.random_integers(0, 10, size=100))
    >> batch = torch.sort(batch).values
    >> sample_batch(batch, max_num_events=2)
    """
    subset = []
    subset_batch_idx = []
    for i in torch.unique(batch_idx):
        batch_idx_i = torch.nonzero(torch.eq(batch_idx, i)).flatten()
        # sample_idx_i = torch.randperm(batch_idx_i.numel())[:num_samples]
        # subset.append(batch_idx_i[sample_idx_i])
        sample_idx_i = batch_idx_i[:num_samples]
        subset.append(sample_idx_i)
        subset_batch_idx.append(torch.ones(sample_idx_i.numel()) * i)
    return torch.cat(subset).long(), torch.cat(subset_batch_idx).long()


def create_and_run_model(dm, num_events: int, index: int, device: torch.device, args: argparse.Namespace, **kwargs):
    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)

    trainloader = dm.train_dataloader()


    #Image size used to initialize the model's layers
    
    #to use ncaltech
    input_shape = [240, 180, 3]

    #to use ncars
    # input_shape = [120, 100, 3]

    input_shape_tensor = torch.Tensor(input_shape)
    
    model = aegnn.models.networks.GraphRes(dm.name,input_shape_tensor, dm.num_classes, pooling_size=args.pooling_size)

    model.to(device)

    model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001,)
   

    for epoch in range(50):

        running_loss = 0.0

        for i, batch in enumerate(trainloader):

            batch_idx = getattr(batch, 'batch')


            subset, subset_batch_idx = sample_batch(batch_idx, num_samples=num_events)

            is_in_subset = torch.zeros(batch_idx.numel(), dtype=torch.bool)
            is_in_subset[subset] = True

            edge_index, edge_attr = subgraph(is_in_subset, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)
            sample = Batch(x=batch.x[is_in_subset, :], pos=batch.pos[is_in_subset, :], y=batch.y,
                        edge_index=edge_index, edge_attr=edge_attr, batch=subset_batch_idx)
            logging.debug(f"Done data-processing, resulting in {sample}")


            sample = sample.to(device)

            optimizer.zero_grad()

            outputs = model.forward(sample)

            loss = criterion(outputs,sample.y)
            loss.backward()
            optimizer.step()

      
            running_loss += loss.item()

            if i % 500 == 499: 
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / (500) :.3f}')
                running_loss = 0.0

         
        #to save the model
        torch.save(model.state_dict(), 'model' + str(epoch) + '_.pt')

    print('Finished Training')

    return model


##################################################################################################
# Experiments ####################################################################################
##################################################################################################
def run_experiments(dm, args, experiments: List[int], num_trials: int, device: torch.device, **model_kwargs
                    ) -> pd.DataFrame:


    runs = list(itertools.product(experiments, list(range(num_trials))))
    # print("runs",runs)
    for num_events, exp_id in tqdm(runs):

        #function which adds the data to graphs aysnchornously
        model = create_and_run_model(dm, num_events, index=exp_id, args=args, device=device, **model_kwargs)

if __name__ == '__main__':
    arguments = parse_args()
    if arguments.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    #to use ncars
    # data_module = aegnn.datasets.NCars(batch_size=64, shuffle=True)

    #to use ncaltech
    data_module = aegnn.datasets.NCaltech101(batch_size=4, shuffle=True)
    data_module.setup()

    #Number of nodes in the graph 5000 for NCaltech101, 1000 for NCars
    event_counts = [5000]
    run_experiments(data_module, arguments, experiments=event_counts, num_trials=1,
                    device=torch.device(arguments.device), log_flops=True, log_runtime=True)
