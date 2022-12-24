import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
import time, os
from math import comb
from random import uniform, seed, choice
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
from copy import deepcopy
from itertools import combinations
from utils import *

class IMGraph:
    def __init__(
        self,
        file_path: str = "", 
        G_nx = None,
        p:float = .5,
        mc:int = 1000,
        eps:float = 0.3,
        l:int = 1,
        max_k = 5,
        k_step = 1,
        directed = False
    ) -> None:
        self.file_path = file_path
        self.p = p
        self.mc = mc
        self.method_spread_map = {}
        self.method_seed_map = {}
        self.method_time_map = {}
        self.method_seed_idx = {}
        self.properties = None
        self.k = max_k
        self.eps = eps
        self.l = l
        self.k_list = [i for i in range(1, max_k+1, k_step)]
        self.directed = directed
        
        self.theta_tim_list = []
        self.theta_imm_list = []
        self.theta_sketch_list = []

        self.have_run_imm = False
        self.have_run_tim = False
        try:
            if self.file_path == "" and G_nx is not None:
                self.G_nx = G_nx
            else:
                self.G_nx = self.load_G_nx(directed)
            
        except Exception as e:
            print("Failed to load the graph by networkx")
            print("Error:")
            print(e)
        try:
            self.G = ig.Graph.from_networkx(self.G_nx)
        except Exception as e:
            print("Failed to load the graph by networkx")
            print("Error:")
            print(e)
        self.n = self.G.vcount()
        self.m = self.G.ecount()
        return
    
    def test(self) -> None:
        print(self.file_path)
    
    def set_p(self, p) -> None:
        self.p = p
        return
    
    def load_G_nx(self, directed = False):
        _create_using = nx.DiGraph if directed else nx.Graph
        if self.file_path.endswith("gml"):
            return nx.read_gml(self.file_path)
        elif self.file_path.endswith("mtx"):
            return read_mtx(self.file_path, create_using=_create_using, skip=2)
        elif self.file_path.endswith("edges"):
            try:
                return nx.read_edgelist(self.file_path, create_using=_create_using)
            except:
                return nx.read_edgelist(self.file_path, data=[("attr_" + str(i), float) for i in range(get_num_col(self.file_path) - 2)], create_using=_create_using)
        print("Cannot process such a file format")
        return None

    # Independent cascade model
    # Used to compute influence spread
    def IC(self, S) -> float:
        """
        Input:
            G: igraph Graph
            S: seed set
            p: probability threshold
            mc: number of MC simulations
        Output:
            average number of influenced nodes
        """
        spread = []     # The number of influenced nodes, starting from S
        # Loop for MC simulations
        for i in range(self.mc):
            # new_active: Newly activated nodes
            # A: all activated nodes
            new_active, A = S[:], S[:]

            # while there are newly added nodes
            while len(new_active) > 0:
                new_ones = []
                # For every newly activated nodes
                for node in new_active:
                    # Determine neighbors that become influenced
                    # np.random.seed(i+5001)       # set random seed
                    # sampling
                    success = np.random.uniform(0,1,len(self.G.neighbors(node,mode="out"))) < self.p
                    # newly activated nodes
                    new_ones += list(np.extract(success, self.G.neighbors(node,mode="out")))
                # compute the newly activated nodes
                new_active = list(set(new_ones) - set(A))
                
                # Add newly activated nodes to the set of activated nodes
                A += new_active
            # number of all activated nodes in this instance
            # print(i, len(A))
            spread.append(len(A))
        return np.mean(spread)
    
    def brute_force(self) -> None:
        SPREAD = []
        combs = combinations(range(self.n), self.k)
        max_spread = 0
        max_seeds = []
        for c in tqdm(combs):
            cur_res = self.IC(list(c))
            if cur_res > max_spread:
                max_spread = cur_res
                max_seeds = c
        SPREAD.append(max_spread)
        self.method_spread_map["EXACT"] = SPREAD
        self.method_seed_idx["EXACT"] = max_seeds

    def proxy(self, proxy="pagerank") -> None:
        st_time = time.time()
        
        Q = zip(range(self.G.vcount()), getattr(self.G, proxy)())
        Q = sorted(Q, key = lambda x: x[1], reverse=True)

        SEED = [Q[i][0] for i in range(self.k)]

        self.method_time_map[proxy] = [time.time() - st_time]*len(self.k_list)

        spread = [self.IC(SEED[:i]) for i in self.k_list]
        self.method_spread_map[proxy] = spread
        self.method_seed_idx[proxy] = SEED
        self.method_seed_map[proxy] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
        return
    
    def run_proxy_methods(self) -> None:
        for metric in proxy_metrics:
            self.proxy(metric)
        return
    # Greedy Algorithm
    def run_greedy(self) -> None:
        """
        Input:
            G: igraph Graph
            k: size of seed set
            p: threshold
            mc: number of mc simulation
        Output:
            S: solution seed set
            spread: number of influenced vertices
        """
        SEED, spread, timelapse, start_time = [], [], [], time.time()
        # loop for k nodes selection
        for _ in tqdm(range(self.k)):
            best_spread = 0    # initialization 
            # for every node that is not in S
            for j in set(range(self.G.vcount())) - set(SEED):
                s = self.IC(SEED+[j])
                if s > best_spread:
                    best_spread, node = s, j
            SEED.append(node)

            # Estimated spread and elapsed time
            spread.append(best_spread)
            timelapse.append(time.time() - start_time)
        self.method_spread_map["greedy"] = spread
        self.method_time_map["greedy"] = timelapse
        self.method_seed_idx["greedy"] = SEED
        self.method_seed_map["greedy"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]
        return
    
    # Cost Effective Lazy Forward
    def run_celf(self):
        st_time = time.time()       # start time
        # marginal gain for every node
        # spread from every single node
        marg_gain = [self.IC([node]) for node in tqdm(range(self.n))]
        # sort the nodes by marginal gain
        Q = sorted(zip(range(self.n), marg_gain), key=lambda x: x[1], reverse=True)

        # seed set initialization: the first node
        # spread: number of all influenced nodes
        # SPREAD: # influenced nodes list
        S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
        Q, LOOKUPS, timelapse = Q[1:], [self.n], [time.time() - st_time]

        
        for _ in tqdm(range(self.k-1)):
            checked, node_lookup = False, 0
            # till the node with the highest MG does not change
            while not checked:
                node_lookup += 1    # The number of times the spread is computed
                current = Q[0][0]
                # calculate the MG of the current node
                Q[0] = (current, self.IC(S + [current]) - spread)
                Q = sorted(Q, key=lambda x: x[1], reverse=True)

                # if new MG is still the highest, exit the loop
                if Q[0][0] == current:
                    checked = True
                
            spread += Q[0][1]
            S.append(Q[0][0])
            SPREAD.append(spread)
            LOOKUPS.append(node_lookup)
            timelapse.append(time.time()-st_time)

            Q = Q[1:]
        self.method_spread_map["CELF"] = SPREAD
        self.method_time_map["CELF"] = timelapse
        self.method_seed_idx["CELF"] = S
        self.method_seed_map["CELF"] = [self.G.vs[idx]["_nx_name"] for idx in S]
        return
    
    def get_RRS(self) -> list:
        """
        Inputs:
            G: igraph Graph
            p: Propagation probability
        """
        source = choice(self.G.vs.indices)
        # mask = np.random.uniform(0, 1, len(self.G.neighbors(source,mode="out"))) < self.p
        dir_G = self.G.copy()
        if not self.directed:
            dir_G.to_directed()
            samp_G = np.array(dir_G.get_edgelist())[np.random.uniform(0, 1, self.m*2) < self.p]
        else:
            samp_G = np.array(dir_G.get_edgelist())[np.random.uniform(0, 1, self.m) < self.p]

        new_nodes, RRS0 = [source], [source]
        while new_nodes:
            tmp = [edge for edge in samp_G if edge[1] in new_nodes]
            tmp = [edge[0] for edge in tmp]
            RRS = list(set(RRS0+tmp))

            new_nodes = list(set(RRS) - set(RRS0))  # New nodes in the RR set

            RRS0 = RRS
        return RRS
    
    def run_RIS(self, k, R = [], num_mc=None) -> None:
        st_time = time.time()
        if num_mc is None:
            eps = self.eps
            c = 4*(1 + eps)*(1+1/k)
            num_mc = int(np.ceil(c*self.m*k*np.log(self.n) / np.power(eps, 2)))
        cur_len = len(R)
        print("Number of MC simulations for RIS:", num_mc)
        for _ in range(cur_len, num_mc):
            R.append(self.get_RRS())
        
        SEED = self.node_selection(R, k)
        return SEED, R
    
    def run_RIS_for_all_k(self):
        SEED = None
        timelapse = []
        all_R = []
        st = time.time()
        for _k in range(1, self.k + 1):
            SEED, all_R = self.run_RIS(_k, R=all_R)
            timelapse.append(time.time() - st)
        if SEED != None and len(SEED) == self.k:
            self.method_seed_idx["RIS"] = SEED
            self.method_seed_map["RIS"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]    
            self.method_time_map["RIS"] = timelapse
        self.have_run_tim = True
        return

    def run_TIM(self, k, R=[]):
        n, eps, l = self.n, self.eps, self.l
        
        kpt = self.kpt_estimation(k)
        lam = (8+2*self.eps)*self.n*(l*np.log(n) + log_n_k(n, k) + np.log(2))*np.power(eps, -2)
        theta = int(np.ceil(lam / kpt))
        
        self.theta_tim_list.append(theta)
        print("Number of RR sets for TIM", theta)
        cur_len = len(R)
        for _ in range(cur_len, theta):
            R.append(self.get_RRS())
        SEED, frac = self.node_selection(R, k)
        return SEED, R

    
    def run_TIM_for_all_k(self):
        SEED = None
        timelapse = []
        all_R = []
        st = time.time()
        for _k in range(1, self.k + 1):
            SEED, all_R = self.run_TIM(_k, R=all_R)
            timelapse.append(time.time() - st)
        if SEED != None and len(SEED) == self.k:
            self.method_seed_idx["TIM"] = SEED
            self.method_seed_map["TIM"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]    
            self.method_time_map["TIM"] = timelapse
        self.have_run_tim = True
        return

    def get_w(self, R):
        res = 0
        for node in R:
            self.G.indegree()[node]
        return res
            

    def kpt_estimation(self, k):
        for i in range(1, int(np.ceil(np.log2(self.n)-1))):
            ci = (6*self.l*np.log(self.n) + 6*np.log(np.log2(self.n))) * np.power(2, i)
            _sum = 0
            for j in range(1, int(np.ceil(ci))):
                R = self.get_RRS()
                kappa = 1 - np.power(1 - self.get_w(R)/self.m, k)
                _sum += kappa
            if _sum / ci > 1 / np.power(2, i):
                return self.n*_sum/(2*ci)
        return 1

    
    def estimate_spread(self, method:str) -> None:
        SPREAD = []
        S = self.method_seed_idx[method]
        for i in range(len(S)):
            SPREAD.append(self.IC(S[:i+1]))
        self.method_spread_map[method] = SPREAD
        return
    
    def run_all_methods(self) -> None:
        self.run_proxy_methods()
        self.run_greedy()
        self.run_celf()
        self.run_RIS()
        self.estimate_spread("RIS")
        return


    def node_selection(self, R_sets, k):
        res = []
        R = deepcopy(R_sets)
        origin_len = len(R)
        for _ in range(k):
            flat_list = [item for sublist in R for item in sublist]
            if len(flat_list) == 0:
                break
            seed = Counter(flat_list).most_common()[0][0]
            res.append(seed)
            R = [rrs for rrs in R if seed not in rrs]
        frac = 1 - len(R)/origin_len
        for _ in range(self.k - len(res)):
            res.append(choice(list(set(range(self.n)) - set(res))))
        
        frac = 1 - len(R)/origin_len
        return res, frac

    def run_IMM(self, k):
        n, eps, l = self.n, self.eps, self.l
        alpha = np.sqrt(l*np.log(n) + np.log(2))
        beta = np.sqrt((1 - np.exp(-1)) * (log_n_k(n, k) + np.log(n) + np.log(2)))
        lambda_star = 2*n*np.power((1 - np.exp(-1))*alpha + beta, 2) / np.power(eps, 2)
        LB = 1
        eps_prime = np.sqrt(2)*eps
        lam_prime = (2 + 2*eps_prime / 3) * (log_n_k(n, k) + l*np.log(n) + np.log(np.log2(n))) * n / np.power(eps_prime, 2)

        all_R = []

        for i in range(1, int(np.ceil(np.log2(self.n)-1))):
            x = n/np.power(2, i)
            theta_i = lam_prime / x
            for j in range(0, int(np.ceil(theta_i))):
                all_R.append(self.get_RRS())
            S_i, frac = self.node_selection(all_R, k)
            if n*frac >= (1-eps_prime) * x:
                LB = n*frac / (1 + eps_prime)
                break
        theta = int(np.ceil(lambda_star / LB))
        
        self.theta_imm_list.append(theta)
        print(f"The number of RR sets in IMM, for k={k}:", theta)
        for i in range(1, theta):
            all_R.append(self.get_RRS())
        SEED, fraction = self.node_selection(all_R, k)
        
        return SEED

    def run_IMM_for_all_k(self):
        SEED = None
        timelapse = []
        st = time.time()
        for _k in range(1, self.k + 1):
            SEED = self.run_IMM(_k)
            timelapse.append(time.time() - st)
        if SEED != None and len(SEED) == self.k:
            self.method_seed_idx["IMM"] = SEED
            self.method_seed_map["IMM"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]    
            self.method_time_map["IMM"] = timelapse
        self.have_run_imm = True
        return

    def run_sketch(self, k, epsilon=0.4):
        if self.have_run_imm == False and self.have_run_tim == False:
            print("Please run TIM or IMM first.")
            return None
        n = self.n
        delta_pp = 1
        eps = epsilon
        if self.have_run_imm and self.have_run_tim:
            min_theta = min(self.theta_imm_list[k-1], self.theta_tim_list[k-1])
        else:
            min_theta = self.theta_imm_list[k-1] if self.have_run_imm else self.theta_tim_list[k-1]
        loglog_theta = np.log(np.log(min_theta) / np.log(1/(1-eps)))
        delta = delta_pp + loglog_theta
        
        theta_try = int(np.ceil(12*(delta + k*np.log(n)) / eps**2))
        if theta_try > min_theta:
            print(f"The proposed algorithm is not effective, k={k}.")
            theta_try = min_theta
        else:
            print(f"The algorithm is effective, k={k}. Number of RR sets={theta_try}.")

        self.theta_sketch_list.append(theta_try)
        
        R = [self.get_RRS() for _ in range(theta_try)]
        SEED, frac = self.node_selection(R, k)
        return SEED

    def run_sketch_for_all_k(self):
        SEED = None
        timelapse = []
        st = time.time()
        for _k in range(1, self.k + 1):
            SEED = self.run_sketch(_k)
            timelapse.append(time.time() - st)
        if SEED != None and len(SEED) == self.k:
            self.method_seed_idx["my_method"] = SEED
            self.method_seed_map["my_method"] = [self.G.vs[idx]["_nx_name"] for idx in SEED]    
            self.method_time_map["my_method"] = timelapse
        return



    def get_properties(self) -> None:
        if self.directed:
            tmp_undirected = nx.to_undirected(self.G_nx)
            giant_comp = tmp_undirected.subgraph(sorted(nx.connected_components(tmp_undirected), key=len, reverse=True)[0])
        else:
            giant_comp = self.G_nx.subgraph(sorted(nx.connected_components(self.G_nx), key=len, reverse=True)[0])
        self.properties = {
            "density": nx.density(giant_comp),
            "diameter": nx.diameter(giant_comp),
            "avg_shortest_path_length": nx.average_shortest_path_length(giant_comp),
            "clustering_coefficient": nx.average_clustering(giant_comp),
        }
        return