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
import pytorch_lightning.metrics.functional as pl_metrics
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph
from tqdm import tqdm
import pickle
import torch.nn as nn
import torch.optim as optim 

from typing import Iterable, Tuple

torch.cuda.empty_cache()
import numpy as np
import pandas as pd
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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


def create_and_run_model(dm, num_events: int,model_to_load,  index: int, device: torch.device, args: argparse.Namespace, **kwargs):
    edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)

    val_loader = dm.val_dataloader()

    #Image size used to initialize the model's layers
    
    #to use ncaltech
    input_shape = [240, 180, 3]

    #to use ncars
    # input_shape = [120, 100, 3]

    input_shape_tensor = torch.Tensor(input_shape)
    
    model = aegnn.models.networks.GraphRes(dm.name,input_shape_tensor, dm.num_classes, pooling_size=args.pooling_size)

    model.to(device)

    model = aegnn.asyncronous.make_model_asynchronous(model, args.radius, list(dm.dims), edge_attr, **kwargs)

 
    model.load_state_dict(torch.load(model_to_load))
 
    accuracy = []

    for i, batch in enumerate(val_loader):
        batch_idx = getattr(batch, 'batch')


        subset, subset_batch_idx = sample_batch(batch_idx, num_samples=num_events)
        is_in_subset = torch.zeros(batch_idx.numel(), dtype=torch.bool)
        is_in_subset[subset] = True

        edge_index, edge_attr = subgraph(is_in_subset, batch.edge_index, edge_attr=batch.edge_attr, relabel_nodes=True)
        sample = Batch(x=batch.x[is_in_subset, :], pos=batch.pos[is_in_subset, :], y=batch.y,
                    edge_index=edge_index, edge_attr=edge_attr, batch=subset_batch_idx)
        logging.debug(f"Done data-processing, resulting in {sample}")

        sample = sample.to(device)

        outputs_i = model.forward(sample)
        y_hat_i = torch.argmax(outputs_i, dim=-1)

        accuracy_i = pl_metrics.accuracy(preds=y_hat_i, target=sample.y).cpu().numpy()
        accuracy.append(accuracy_i)


    
    return float(np.mean(accuracy))


def run_experiments(dm, args, model_to_load, pkl_file, csv_file, experiments: List[int], num_trials: int, device: torch.device, **model_kwargs
                    ) -> pd.DataFrame:


    runs = list(itertools.product(experiments, list(range(num_trials))))

    df = pd.DataFrame()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "aegnn_results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, pkl_file)

    #Validate with graphs containing increasing number of Nodes in each validation.
    max_num_events = [2000,3000,4000, 5000]
    
    for sample_size_nodes in max_num_events:
        for num_events, exp_id in tqdm(runs):

            accuracy = create_and_run_model(dm, sample_size_nodes, model_to_load, index=exp_id, args=args, device=device, **model_kwargs)

            logging.debug(f"Evaluation with max_num_events = {sample_size_nodes} => Recognition accuracy = {accuracy}")

            df = df.append({"accuracy": accuracy, "max_num_events": sample_size_nodes}, ignore_index=True)
            df.to_pickle(output_file)

        print(f"Results are logged in {output_file}")

    file_name = os.path.join(output_dir, csv_file)

    with open(file_name, 'rb') as f:
        data = pickle.load(f)

        data.to_csv(file_name, sep='\t')
    return df


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.debug:
        _ = aegnn.utils.loggers.LoggingLogger(None, name="debug")

    #to use ncars
    # data_module = aegnn.datasets.NCars(batch_size=64, shuffle=True)

    #to use ncaltech
    data_module = aegnn.datasets.NCaltech101(batch_size=4, shuffle=True)
    data_module.setup()

    #over-written
    event_counts = [0]

    #validate only these checkpoints
    epochs_to_validate = [4,9,14,19,24,29,34,39,44,49]


    for epochs in epochs_to_validate:
     

        model_to_load = "model" + str(epochs) +  "_.pt"

        pkl_file = "model" + str(epochs) +  "_.pkl"

        csv_file = "model" + str(epochs) +  "_.csv"

        run_experiments(data_module, arguments, model_to_load, pkl_file, csv_file, experiments=event_counts, num_trials=1,
                        device=torch.device(arguments.device), log_flops=True, log_runtime=True)
