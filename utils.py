import networkx as nx
import numpy as np
import pickle
proxy_metrics = ["pagerank", "outdegree", "betweenness"]

def read_mtx(path,
    skip = 0,
    comments="#",
    delimiter=None,
    create_using=None,
    nodetype=None,
    data=True,
    edgetype=None,
    encoding="utf-8",):
    """
    Inputs:
        path: file path
        skip: the number of lines that should be skipped
    Outputs:
        networkx graph
    """
    with open(path, 'r') as f:
        str_list = f.readlines()
    return nx.parse_edgelist(
        str_list[skip:],
        comments=comments,
        delimiter=delimiter,
        create_using=create_using,
        nodetype=nodetype,
        data=data,
    )

def get_num_col(file_name):
    with open(file_name, "rb") as f:
        line = next(f)
        return len(line.decode("utf-8").split(' '))

def save_instance(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    
def load_instance(file_path):
    with open(file_path, "rb") as f:
        res = pickle.load(f)
    return res

def log_n_k(n, k):
    res = 0
    for i in range(1, k+1):
        res += np.log((n - i + 1) / i)
    return res