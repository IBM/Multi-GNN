import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor
import numpy as np
import pandas as pd
import pickle

def to_adj_nodes_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1) if not isinstance(data, HeteroData) else torch.cat((data['node', 'to', 'node'].edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out

def to_adj_edges_with_times(data):
    num_nodes = data.num_nodes
    timestamps = torch.zeros((data.edge_index.shape[1], 1)) if data.timestamps is None else data.timestamps.reshape((-1,1))
    edges = torch.cat((data.edge_index.T, timestamps), dim=1)
    # calculate adjacent edges with times per node
    adj_edges_out = dict([(i, []) for i in range(num_nodes)])
    adj_edges_in = dict([(i, []) for i in range(num_nodes)])
    for i, (u,v,t) in enumerate(edges):
        u,v,t = int(u), int(v), int(t)
        adj_edges_out[u] += [(i, v, t)]
        adj_edges_in[v] += [(i, u, t)]
    return adj_edges_in, adj_edges_out

def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def time_deltas(data, adj_edges_list):
    time_deltas = torch.zeros(data.edge_index.shape[1], 1)
    if data.timestamps is None:
        return time_deltas
    for v, edges in adj_edges_list.items():
        if len(edges) < 1: continue
        a = np.array(edges)
        a = a[a[:, -1].argsort()]
        a_tds = [0] + [a[i+1,-1] - a[i,-1] for i in range(a.shape[0]-1)]
        tds = np.hstack((a[:,0].reshape(-1,1), np.array(a_tds).reshape(-1,1)))
        for i,td in tds:
            time_deltas[i] = td
    return time_deltas

class GraphData(Data):
    '''This is the homogenous graph object we use for GNN training if reverse MP is not enabled'''
    def __init__(
        self, x: OptTensor = None, edge_index: OptTensor = None, edge_attr: OptTensor = None, y: OptTensor = None, pos: OptTensor = None, 
        readout: str = 'edge', 
        num_nodes: int = None,
        timestamps: OptTensor = None,
        node_timestamps: OptTensor = None,
        **kwargs
        ):
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        self.readout = readout
        self.loss_fn = 'ce'
        self.num_nodes = int(self.x.shape[0])
        self.node_timestamps = node_timestamps
        if timestamps is not None:
            self.timestamps = timestamps  
        elif edge_attr is not None:
            self.timestamps = edge_attr[:,0].clone()
        else:
            self.timestamps = None

    def add_ports(self):
        reverse_ports = True
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self.edge_index, adj_list_in)
        out_ports = [ports(self.edge_index.flipud(), adj_list_out)] if reverse_ports else []
        self.edge_attr = torch.cat([self.edge_attr, in_ports] + out_ports, dim=1)
        return self

    def add_time_deltas(self):
        reverse_tds = True
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = [time_deltas(self, adj_list_out)] if reverse_tds else []
        self.edge_attr = torch.cat([self.edge_attr, in_tds] + out_tds, dim=1)
        return self

class HeteroGraphData(HeteroData):
    '''This is the heterogenous graph object we use for GNN training if reverse MP is enabled'''
    def __init__(
        self,
        readout: str = 'edge',
        **kwargs
        ):
        super().__init__(**kwargs)
        self.readout = readout

    @property
    def num_nodes(self):
        return self['node'].x.shape[0]
        
    @property
    def timestamps(self):
        return self['node', 'to', 'node'].timestamps

    def add_ports(self):
        adj_list_in, adj_list_out = to_adj_nodes_with_times(self)
        in_ports = ports(self['node', 'to', 'node'].edge_index, adj_list_in)
        out_ports = ports(self['node', 'rev_to', 'node'].edge_index, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_ports], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_ports], dim=1)
        return self

    def add_time_deltas(self):
        adj_list_in, adj_list_out = to_adj_edges_with_times(self)
        in_tds = time_deltas(self, adj_list_in)
        out_tds = time_deltas(self, adj_list_out)
        self['node', 'to', 'node'].edge_attr = torch.cat([self['node', 'to', 'node'].edge_attr, in_tds], dim=1)
        self['node', 'rev_to', 'node'].edge_attr = torch.cat([self['node', 'rev_to', 'node'].edge_attr, out_tds], dim=1)
        return self
    
def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def create_hetero_obj(x,  y,  edge_index,  edge_attr, timestamps, args):
    '''This function creates a heterogenous graph object for reverse message passing'''
    data = HeteroGraphData()

    data['node'].x = x
    data['node', 'to', 'node'].edge_index = edge_index
    data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
    data['node', 'to', 'node'].edge_attr = edge_attr
    data['node', 'rev_to', 'node'].edge_attr = edge_attr
    if args.ports:
        #swap the in- and outgoing port numberings for the reverse edges
        data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]
    data['node', 'to', 'node'].y = y
    data['node', 'to', 'node'].timestamps = timestamps
    
    return data

def get_dict_val(name, collection):
    if name in collection:
        val = collection[name]
    else:
        val = len(collection)
        collection[name] = val
    return val

def preprocess_regenerated_data(transaction_file, args):
    """Pre-processes the regenerated AML data to be used with the existing gnn scripts.
     Most importantly, reindexes the nodes and edges to be used by a GNN.

    Args:
      transaction_file: The csv file where the transaction information is stored.
      args: The command line arguments.

    Returns:
      df_edges: A pandas dataframe containing all the edge features.
      df_nodes: A pandas dataframe containing all the node features.
    """

    # Load the dataframe
    #TODO: replace the path here with your actual data path
    transaction_file = f"/path_to_data/{args.data}/transactions_formatted.txt"
    df = pd.read_csv(transaction_file)

    # Rename and reorder the columns to get the same format as the original files
    # from_id and to_id now contain the holding ids
    df.columns = ["Timestamp", "From Bank", "From ID", "To Bank", "To ID", "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    df = df[["From ID", "To ID", "Timestamp", "Amount Paid", "Payment Currency", "Amount Received", "Receiving Currency", "Payment Format", "Is Laundering"]]

    # Redo the time column to get unix time
    df.sort_values(by='Timestamp', inplace=True)
    # 2. Find the minimum Unix time in the column and subtract 990 seconds
    min_time = df['Timestamp'].min() - 10
    # 3. Subtract this value from every entry in the 'Timestamp' column
    df['Timestamp'] = df['Timestamp'] - min_time

    # Create a dictionary that maps all nodes holding ids to their respective local id
    # n_id_map has all holding ids as keys and their local id counterpart as the value
    node_ids = np.unique(df[["From ID", "To ID"]].values)
    n_id_map = {value: index for index, value in enumerate(node_ids)}
    df["From ID"] = df["From ID"].map(n_id_map)
    df["To ID"] = df["To ID"].map(n_id_map)

    # Categorically encode the neccessary features
    currency = dict()
    paymentFormat = dict()
    df["Payment Currency"] = df["Payment Currency"].apply(lambda x: get_dict_val(x, currency))
    df["Receiving Currency"] = df["Receiving Currency"].apply(lambda x: get_dict_val(x, currency))
    df["Payment Format"] = df["Payment Format"].apply(lambda x: get_dict_val(x, paymentFormat))

    # Final edge df
    df_edges = df.copy()

    # Load the node embeddings
    # To use this, you need embeddings pre-computed from Tab-GT
    #TODO: replace the path here with your actual data path
    with open(f'/path_to_data/{args.data}/{args.node_feats[0]}_embeddings.pkl', 'rb') as handle:
        embeddings = pickle.load(handle)
    

    df_nodes = pd.DataFrame.from_dict(embeddings, orient='index').reset_index()
    df_nodes.rename(columns={'index':'Holding ID'}, inplace=True)
    df_edges.rename(columns={'Receiving Currency':'Received Currency', 'From ID': 'from_id', 'To ID': 'to_id'}, inplace=True)

    df_nodes['NodeID'] = df_nodes['Holding ID'].map(n_id_map)
    df_nodes = df_nodes.sort_values(by=['NodeID'])

    if args.new_no_feats:
        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})

    return df_edges, df_nodes