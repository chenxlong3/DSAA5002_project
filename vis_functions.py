from IMGraph import IMGraph
import matplotlib.pyplot as plt

method_marker = {
    "pagerank": 'o',
    "outdegree": "*",
    "betweenness": "x",
    "greedy": "+",
    "CELF": "s",
    "RIS": "D",
    "TIM": "p",
    "IMM": "h",
    "my_method": "P"
}
font = {'family': 'serif',
        'weight': 'normal',
        }
label_size = 15

def vis_methods(IM_G: IMGraph, **kwargs) -> None:
    fig_size = (9, 3)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    
    fig = plt.figure(figsize=fig_size)
    plt.subplot(121)
    for method in IM_G.method_spread_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_spread_map[method], marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)
    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=label_size)
    plt.ylabel("Expected Spread", size=label_size, fontdict=font)

    plt.subplot(122)
    for method in IM_G.method_spread_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_time_map[method], marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)

    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=label_size)
    plt.ylabel("Time Cost (seconds)", size=label_size, fontdict=font)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(IM_G.method_spread_map.keys()), prop={"family": "serif"}, bbox_to_anchor=[0.5, 1.1], frameon=False)
    plt.tight_layout()
    
    if "save_to_path" in kwargs.keys():
        plt.savefig(kwargs["save_to_path"], bbox_inches='tight')

    plt.show()

def vis_feature_delta(feature_names, feature_data, delta_list, line=False, **kwargs):
    fig_size = (12, 2.5)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    plt.figure(figsize=fig_size)
    for i, name, data in zip(range(len(feature_names)), feature_names, feature_data):
        plt.subplot(1, 4, i+1)
        if line:
            plt.plot(data, delta_list)
        else:
            plt.scatter(data, delta_list)
        plt.xlabel(name, fontdict=font)
        plt.ylabel(r"$\delta$", fontdict=font)
    plt.tight_layout()
    if "save_to_path" in kwargs.keys():
        plt.savefig(kwargs["save_to_path"], bbox_inches='tight')
    plt.show()

# Deprecated
def vis_methods_spread(IM_G:IMGraph, **kwargs) -> None:
    fig_size = (4, 3)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    plt.figure(figsize=fig_size)

    for method in IM_G.method_spread_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_spread_map[method],  marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)
        
    plt.legend()
    # plt.title("Resulting Influence Spread of Different Methods", size=15, fontdict=font)
    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=label_size)
    plt.ylabel("Expected Spread", size=label_size, fontdict=font)
    if "save_to_path" in kwargs.keys():
        plt.savefig(kwargs["save_to_path"])
    plt.show()

# Deprecated
def vis_methods_time(IM_G:IMGraph, **kwargs) -> None:
    fig_size = (4, 3)
    if "figsize" in kwargs.keys():
        fig_size= kwargs["figsize"]
    plt.figure(figsize=fig_size)

    for method in IM_G.method_time_map.keys():
        plt.plot(IM_G.k_list, IM_G.method_time_map[method],  marker=method_marker[method], label=method, markersize=8, mfc="None", linewidth=1)
        
    plt.legend()
    # plt.title("Time Cost of Different Methods", size=label_size, fontdict=font)
    plt.xticks(IM_G.k_list)
    plt.xlabel(r"$k$", size=15)
    plt.ylabel("Time Cost (seconds)", size=label_size, fontdict=font)
    if "save_to_path" in kwargs.keys():
        plt.savefig(kwargs["save_to_path"])
    plt.show()