import numpy as np
from sklearn.preprocessing import scale
import os
import itertools

#Function borrowed from https://github.com/BorgwardtLab/WWL
def retrieve_graph_filenames(data_directory):
    # Load graphs
    files = os.listdir(data_directory)
    graphs = [g for g in files if g.endswith('gml')]
    graphs.sort()
    return [os.path.join(data_directory, g) for g in graphs]

#Function borrowed from https://github.com/BorgwardtLab/WWL
def load_continuous_graphs(data_directory):
    graph_filenames = retrieve_graph_filenames(data_directory)

#Custom function to implement Continous WL for continuous labeled graphs
def CWL_continous_labels(node_features, adj_mat, h,n_nodes):

    n_graphs = len(node_features)
    labels_sequence = []

    for n in range(len(n_nodes)):
     
        nodes_list = []
        for each in range(n_nodes[n]):
            nodes_list.append(each)

        n_values = np.max(nodes_list) + 1

        one_hot_list = np.eye(n_values)[nodes_list]

        initial_one_hot = one_hot_list.copy()

        graph_feat = []
        
        for i in range(h+1):
            if i == 0:
               
                one = np.ones_like(adj_mat[n])
                start = np.dot(one,node_features[n])
 
                graph_feat.append(np.dot(one,node_features[n]))
            
            else:
                #Aggregation
                like_create_adj_avg = (np.dot(adj_mat[n],graph_feat[i-1]))

                #Matrix M
                unique_matrix = np.random.randn(*like_create_adj_avg.shape)

                #Hash the aggregation using M
                random_feature = np.multiply(like_create_adj_avg, unique_matrix)

                #New feature
                final_feature = random_feature + graph_feat[i-1]

                graph_feat.append(final_feature)

        labels_sequence.append(np.concatenate(graph_feat, axis = 1))
        
        if i % 100 == 0:
            print(f'Processed {i} graphs out of {n_graphs}')

    return labels_sequence

#Function to be used for Graphs with Continuous node features
def compute_CWL_continuous(data_directory, h):

    node_features, adj_mat, n_nodes = load_continuous_graphs(data_directory)

    node_features_data = scale(np.concatenate(node_features, axis=0), axis = 0)
    splits_idx = np.cumsum(n_nodes).astype(int)
    node_features_split = np.vsplit(node_features_data,splits_idx)		
    node_features = node_features_split[:-1]

    labels_sequence = CWL_continous_labels(node_features, adj_mat, h,n_nodes)

    return labels_sequence


#Function to be used for Graphs with Discrete node features
def compute_wl_embeddings_discrete_ours(data_directory, h):

    node_features, adj_mat, n_nodes = load_continuous_graphs(data_directory)

    #set size of the to be created Continuous label
    label_size = 2
    

    label_sequences = [
        np.full(((node*label_size), h+1), np.nan) for node in n_nodes
    ]   
    

    for n in range(len(n_nodes)):

        nodes_list = []
        for each in range(n_nodes[n]):
            nodes_list.append(each)

        n_values = np.max(nodes_list) + 1

        #Start off as the labels to be a one-hot vector
        one_hot_list = np.eye(n_values)[nodes_list]

        initial_one_hot = one_hot_list.copy()

        for i in range(h+1):
            #Matrix M
            unique_matrix = np.random.normal(loc=0, scale=1.0,size=(n_nodes[n], label_size))

            #Aggregation
            initial_one_hot = (np.dot(initial_one_hot, adj_mat[n])).tolist()

            #Hash the aggregation using M
            label = (np.dot(initial_one_hot, unique_matrix))

            #Normalize labels
            norm_label = (label / np.linalg.norm(label)).tolist()

            new_label = list(itertools.chain(*norm_label))

            label_sequences[n][:,i] = new_label
       
    return label_sequences
