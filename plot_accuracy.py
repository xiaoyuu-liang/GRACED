import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
import argparse

datasets = ['mutag', 'imdb', 'nci1', 'proteins']
seed = [42, 0, 78, 71]
naive_acc = [0.84, 0.77, 0.8175, 0.75]

flip_prob_X_mutag = ['0.00/0.00', '0.09/0.01', '0.30/0.05', '0.57/0.09', '0.68/0.11']
flip_prob_X_imdb = ['0.00/0.00', '0.10/0.00', '0.35/0.00', '0.65/0.00', '0.79/0.01']
flip_prob_X_nci1 = ['0.00/0.00', '0.10/0.00', '0.35/0.01', '0.65/0.02', '0.78/0.02']
flip_prob_X_proteins = ['0.00/0.00', '0.06/0.03', '0.24/0.12', '0.44/0.22', '0.53/0.27']

flip_prob_E_mutag = ['0.00/0.00', '0.09/0.01', '0.31/0.04', '0.58/0.08', '0.70/0.10']
flip_prob_E_imdb = ['0.00/0.00', '0.06/0.04', '0.21/0.14', '0.39/0.27', '0.47/0.33']
flip_prob_E_nci1 = ['0.00/0.00', '0.09/0.01', '0.31/0.02', '0.58/0.04', '0.70/0.05']
flip_prob_E_proteins = ['0.00/0.00', '0.09/0.01', '0.34/0.02', '0.63/0.03', '0.77/0.03']


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
    [0.74, 0.71, 0.78, 0.67, 0.65], # adj 0
    [0.76, 0.72, 0.69, 0.73, 0.64], # adj 100
    [0.75, 0.75, 0.77, 0.75, 0.67], # adj 200
    [0.75, 0.74, 0.70, 0.63, 0.55], # adj 300
    [0.69, 0.67, 0.67, 0.66, 0.58], # adj 350
]
# graphsage cider 300 300 0.75
sparse_imdb_acc = [
# attr  0,   100,  200,  300,  350
    [0.68, 0.70, 0.73, 0.66, 0.51], # adj 0
    [0.62, 0.64, 0.55, 0.67, 0.57], # adj 100
    [0.67, 0.72, 0.75, 0.67, 0.65], # adj 200
    [0.63, 0.52, 0.66, 0.49, 0.45], # adj 300
    [0.68, 0.51, 0.55, 0.62, 0.56], # adj 350
]

cider_nci1_acc = [
# attr  0,   100,  200,  300,  350
    [0.6253, 0.6302, 0.6058, 0.5912, 0.6131], # adj 0
    [0.6326, 0.6326, 0.6131, 0.5864, 0.6058], # adj 100
    [0.6302, 0.6301, 0.6034, 0.5620, 0.5863], # adj 200
    [0.6204, 0.6107, 0.5864, 0.5815, 0.6010], # adj 300
    [0.5912, 0.6010, 0.5985, 0.6229, 0.6375], # adj 350
]

sparse_nci1_acc = [
# attr  0,   100,  200,  300,  350  
    [0.4793, 0.5158, 0.5036, 0.4817, 0.5255], # adj 0
    [0.4963, 0.4890, 0.5060, 0.5109, 0.5109], # adj 100
    [0.5182, 0.4866, 0.5085, 0.5109, 0.5109], # adj 200
    [0.5182, 0.4866, 0.4647, 0.4622, 0.5085], # adj 300
    [0.5401, 0.5523, 0.5255, 0.5596, 0.5231], # adj 350
]

cider_proteins_acc = [
# attr  0,   100,  200,  300,  350 
    [0.6517, 0.6339, 0.5714, 0.5625, 0.5625], # adj 0
    [0.6160, 0.5804, 0.5982, 0.5714, 0.5982], # adj 100
    [0.6428, 0.6429, 0.6429, 0.6250, 0.6250], # adj 200
    [0.7054, 0.6964, 0.7142, 0.6964, 0.6964], # adj 300
    [0.6786, 0.6964, 0.6786, 0.6696, 0.6964], # adj 350
]

sparse_proteins_acc = [
# attr  0,   100,  200,  300, 350
    [0.6250, 0.5803, 0.5446, 0.5178, 0.5535], # adj 0
    [0.5625, 0.5982, 0.6428, 0.5714, 0.6428], # adj 100
    [0.6428, 0.6339, 0.6607, 0.5803, 0.6250], # adj 200
    [0.5892, 0.6875, 0.5982, 0.5625, 0.6071], # adj 300
    [0.5714, 0.6071, 0.6964, 0.6250, 0.5803], # adj 350
]

def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--data", type=str, required=True,help="dataset name")
    arg.add_argument("--gap", type=str, required=False, default='sparse', help="gap type")

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
    colors = ["#ffffff", "#cdcfe1", "#9193b4", "#5c6592"]
    n_bins = 200  # Increase this number for a smoother transition between colors
    cmap_name = "MintCmap"

    # Create the colormap
    mint_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Draw heatmap
    plt.figure(figsize=(10.5, 8.5))
    ax = sns.heatmap(acc_gap, annot=True, cmap=mint_cmap, fmt=".2f", 
                     xticklabels=flip_prob_X, yticklabels=flip_prob_E, annot_kws={"size":14},
                     vmin=0.0, vmax=0.25,
                     cbar_kws={'ticks': np.linspace(0.0, 0.25, 6), 'pad':0.02, 'shrink': 1, 'aspect': 30})
    plt.xlabel("X Flip Probability", fontsize=30, labelpad=20)
    plt.ylabel("A Flip Probability", fontsize=30, labelpad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.gca().invert_yaxis()
    # Set color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Clean accuracy gap', size=30)
    cbar.ax.yaxis.set_tick_params(labelsize=22)
    cbar.ax.tick_params(labelsize=22)
    
    plt.minorticks_off()
    plt.savefig(f"figs/{gap}_{dataset}.png")
    print(f"figs/{gap}_{dataset}.png saved")
    plt.show()
    
    return
    

if __name__ == "__main__":
    
    main()

