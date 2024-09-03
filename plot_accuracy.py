import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import argparse

datasets = ['mutag', 'imdb', 'nci1', 'proteins']
seed = [42, 0, 78, 71]
naive_acc = [0.8421, 0.0, 0.0, 0.7589]

flip_prob_X_mutag = [0.00, 0.09, 0.30, 0.57, 0.68]
flip_prob_X_imdb = [0.00, 0.10, 0.35, 0.65, 0.79]
flip_prob_X_nci1 = [0.00, 0.10, 0.35, 0.65, 0.78]
flip_prob_X_proteins = [0.00, 0.06, 0.24, 0.44, 0.]

flip_prob_E_mutag = [0.00, 0.09, 0.31, 0.58, 0.70]
flip_prob_E_imdb = [0.00, 0.09, 0.31, 0.58, 0.70]
flip_prob_E_nci1 = [0.00, 0.09, 0.31, 0.58, 0.70]
flip_prob_E_proteins = [0.00, 0.09, 0.34, 0.63, 0.77]


cider_mutag_acc = [
# attr  0,   100,  200,  300,  350  
    [0.8421, 0.8421, 0.7895, 0.7895, 0.7895], # adj 0
    [0.8421, 0.7895, 0.7895, 0.7895, 0.7895], # adj 100
    [0.7895, 0.7895, 0.7895, 0.7895, 0.7895], # adj 200
    [0.7895, 0.7895, 0.7895, 0.7895, 0.7895], # adj 300
    [0.7895, 0.7895, 0.7895, 0.7895, 0.7895], # adj 350
]

sparse_mutag_acc = [
# attr  0,   100,  200,  300,  350
    [0.7368, 0.6842, 0.6842, 0.6842, 0.6842], # adj 0
    [0.6842, 0.6842, 0.6842, 0.6842, 0.6842], # adj 100
    [0.6842, 0.6842, 0.6842, 0.6842, 0.6315], # adj 200
    [0.6315, 0.6315, 0.6842, 0.6842, 0.6842], # adj 300
    [0.6842, 0.6842, 0.6842, 0.6842, 0.6315], # adj 350
]

cider_imdb_acc = [
# attr  0,   100,  200,  300,  350 
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

sparse_imdb_acc = [
# attr  0,   100,  200,  300,  350
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

cider_nci1_acc = [
# attr  0,   100,  200,  300,  350  
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

sparse_nci1_acc = [
# attr  0,   100,  200,  300,  350
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

cider_proteins_acc = [
# attr  0,   100,  200,  300,  350 
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

sparse_proteins_acc = [
# attr  0,   100,  200,  300, 350
    [0., 0., 0., 0., 0.], # adj 0
    [0., 0., 0., 0., 0.], # adj 100
    [0., 0., 0., 0., 0.], # adj 200
    [0., 0., 0., 0., 0.], # adj 300
    [0., 0., 0., 0., 0.], # adj 350
]

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, required=True,help="dataset name")
    arg.add_argument("--gap", type=str, required=True, default='', help="gap type")

    args = vars(arg.parse_args())

    return args["data"], args["gap"]

def main():

    dataset, gap = parse_arguments()
    
    if dataset == 'mutag':
        acc_gap_sparse = np.array(cider_mutag_acc) - np.array(sparse_mutag_acc)
        acc_gap_naive = np.array(cider_mutag_acc) - naive_acc[0]
        flip_prob_X = flip_prob_X_mutag
        flip_prob_E = flip_prob_E_mutag
    if dataset == 'imdb':
        acc_gap_sparse = np.array(cider_imdb_acc) - np.array(sparse_imdb_acc)
        acc_gap_naive = np.array(cider_imdb_acc) - naive_acc[1]
        flip_prob_X = flip_prob_X_imdb
        flip_prob_E = flip_prob_E_imdb
    if dataset == 'nci1':
        acc_gap_sparse = np.array(cider_nci1_acc) - np.array(sparse_nci1_acc)
        acc_gap_naive = np.array(cider_nci1_acc) - naive_acc[2]
        flip_prob_X = flip_prob_X_nci1
        flip_prob_E = flip_prob_E_nci1
    if dataset == 'proteins':
        acc_gap_sparse = np.array(cider_proteins_acc) - np.array(sparse_proteins_acc)
        acc_gap_naive = np.array(cider_proteins_acc) - naive_acc[3]
        flip_prob_X = flip_prob_X_proteins
        flip_prob_E = flip_prob_E_proteins
    
    if gap == 'naive':
        acc_gap = acc_gap_naive
    if gap == 'sparse':
        acc_gap = acc_gap_sparse
    print(f'Average accuracy gap with {gap} for {dataset}: {np.mean(acc_gap):.6f}')

    # Define your custom colors
    colors = ["#e5c8a0", "#f1e1c9", "#cdcfe1", "#9193b4", "#5c6592"]
    n_bins = 100  # Increase this number for a smoother transition between colors
    cmap_name = "MintCmap"

    # Create the colormap
    mint_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Draw heatmap
    plt.figure(figsize=(10, 8.5))
    ax = sns.heatmap(acc_gap, annot=True, cmap=mint_cmap, fmt=".2f", 
                     xticklabels=flip_prob_X, yticklabels=flip_prob_E, annot_kws={"size":14},
                     vmin=-0.05, vmax=0.15,
                     cbar_kws={'ticks': np.linspace(-0.05, 0.15, 5)})
    plt.xlabel("X Flip Probability", fontsize=26, labelpad=20)
    plt.ylabel("A Flip Probability", fontsize=26, labelpad=20)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    # Set color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Clean accuracy gap', size=26)
    cbar.ax.yaxis.set_tick_params(labelsize=22)
    cbar.ax.tick_params(labelsize=22)

    plt.savefig(f"figs/{gap}_{dataset}.png")
    print(f"figs/{gap}_{dataset}.png saved")
    plt.show()
    
    return
    

if __name__ == "__main__":
    
    main()

