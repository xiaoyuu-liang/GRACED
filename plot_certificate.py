import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import seaborn as sns
import argparse

# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.01-0.65-E0.00-0.00].pth
# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.00-0.00-E0.00-0.66].pth
# checkpoints/cora/async_pilot/test_only/gcn_cora_[1-X0.01-0.65-E0.00-0.66].pth

# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.00-0.00-E0.00-0.66].pth
# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.01-0.65-E0.00-0.00].pth
# rand_gnn_checkpoints/gcn_cora/gcn_cora_[1-X0.01-0.65-E0.00-0.66].pth

# rand_results/cora/GCN_cora_[0.8-0.01-0.65].pth


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--cert", type=str, required=True,help="path to the CiDer certificate file")
    arg.add_argument("--joint", type=str, required=False, default='', help="joint certificate slice")
    arg.add_argument("--singular", type=str, required=False, default='', help="singular certificate")
    
    args = vars(arg.parse_args())

    joint = args["joint"].split(',') if args["joint"] else []
    joint = (joint[0], int(joint[1]), int(joint[2])) if joint else ()

    return args["cert"], joint, args["singular"]

def main():
    path, joint, singular = parse_arguments()
    
    cert = torch.load(path)['multiclass']['cert_acc']

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    if joint:
        max_ra_adj, max_rd_adj, max_ra_att, max_rd_att = cert[0]
        print(f'max radius for joint certificate: {max_ra_adj, max_rd_adj, max_ra_att, max_rd_att}')
        x_coords_adj, y_coords_adj, x_coords_att, y_coords_att = cert[1]
        cert_acc = cert[2]

        heatmap = np.zeros((6, 17, 8, 17))
        for x_adj, y_adj, x_att, y_att, acc in zip(x_coords_adj, y_coords_adj, x_coords_att, y_coords_att, cert_acc):
            heatmap[x_adj, y_adj, x_att, y_att] = acc
        heatmap[0, 0, 0, 0] = torch.load(path)['majority_acc']
        
        if joint[0] == 'adj':
            heatmap = heatmap[joint[1], joint[2], :, :]
            max_ra, max_rd = max_ra_att, max_rd_att
        elif joint[0] == 'att':
            heatmap = heatmap[:, :, joint[1], joint[2]]
            max_ra, max_rd = max_ra_adj, max_rd_adj
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")
    else:
        max_ra, max_rd = cert[0]
        print(f'max radius for singular certificate: {max_ra, max_rd}')
        x_coords, y_coords = cert[1]
        acc = cert[2]
        heatmap = np.zeros((6, 18))
        for x, y, acc in zip(x_coords, y_coords, acc):
            heatmap[x, y] = acc
        heatmap[0, 0] = torch.load(path)['majority_acc']
    
    # scale_factor = 10**2
    # heatmap = heatmap * scale_factor

    accs = [f'{elem:.2f}' for row in heatmap for elem in row if elem > 1e-3]
    print(accs)

    # Define your custom colors
    if joint:
        if joint[0] == 'adj':
            colors = ["#ffffff", "#dff3f8", "#9bc7df", "#5385bd"]
        elif joint[0] == 'att':
            colors = ["#ffffff", "#ddf3de", "#aadca9", "#519d78"] 
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")
    elif singular == 'adj':
        colors = ["#ffffff", "#ddf3de", "#aadca9", "#519d78"] 
    elif singular == 'att':
        colors = ["#ffffff", "#dff3f8", "#9bc7df", "#5385bd"]
    n_bins = 100  # Increase this number for a smoother transition between colors
    cmap_name = "SkyCmap"
    sky_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    log_heatmap = np.log10(heatmap + 1e-5)

    # truncate
    for col in range(heatmap.shape[1]-1, -1, -1):
        if np.any(heatmap[:, col] >= 1e-5):
            break
    for row in range(heatmap.shape[0]-1, -1, -1):
        if np.any(heatmap[row, :] >= 1e-5):
            break
    
    plt.figure(figsize=(6, 3))
    ax = sns.heatmap(heatmap, cmap=sky_cmap, fmt=".2f", cbar=False, annot=False, annot_kws={"size": 4})
    plt.xticks(np.arange(0.5, heatmap.shape[1], 2))
    plt.yticks(np.arange(0.5, heatmap.shape[0], 2))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.set_xticklabels(np.arange(0, heatmap.shape[1], 2))
    ax.set_yticklabels(np.arange(0, heatmap.shape[0], 2))


    sm = plt.cm.ScalarMappable(cmap=sky_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, format='% .2f')
    cbar.set_label('Certified accuracy', size=14)
    cbar.set_ticks(np.linspace(0, 1, 5))
    cbar.outline.set_visible(False)

    level = 0.2
    color_2 = '#FC9527'
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] >= level:
                if heatmap[i+1, j] < level:
                    plt.step([j, j+1], [i+1, i+1], where='mid', color=color_2, linestyle='--')
                if heatmap[i, j+1] < level:
                    plt.step([j+1, j+1], [i, i+1], where='mid', color=color_2, linestyle='--')

    level = 0.4
    color_4 = '#FC9527'
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] >= level:
                if heatmap[i+1, j] < level:
                    plt.step([j, j+1], [i+1, i+1], where='mid', color=color_4)
                if heatmap[i, j+1] < level:
                    plt.step([j+1, j+1], [i, i+1], where='mid', color=color_4)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_2, linestyle='--', lw=2, label='0.2 Contour'),
        Line2D([0], [0], color=color_4, lw=2, label='0.4 Contour')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12)

    label_fontsize = 16
    label_labelpad = 12
    if joint:
        if joint[0] == 'adj':
            ax.set_xlabel("Budget $\Delta_X^-$", fontsize=label_fontsize, labelpad=label_labelpad)
            ax.set_ylabel("Budget $\Delta_X^+$", fontsize=label_fontsize, labelpad=label_labelpad)
        elif joint[0] == 'att':
            ax.set_xlabel("Budget $\Delta_A^-$", fontsize=label_fontsize, labelpad=label_labelpad)
            ax.set_ylabel("Budget $\Delta_A^+$", fontsize=label_fontsize, labelpad=label_labelpad)
        else:
            raise ValueError("joint certificate slice must be either 'adj' or 'att'")
    elif singular == 'adj':
        ax.set_xlabel("Budget $\Delta_A^-$", fontsize=label_fontsize, labelpad=label_labelpad)
        ax.set_ylabel("Budget $\Delta_A^+$", fontsize=label_fontsize, labelpad=label_labelpad)
    elif singular == 'att':
        ax.set_xlabel("Budget $\Delta_X^-$", fontsize=label_fontsize, labelpad=label_labelpad)
        ax.set_ylabel("Budget $\Delta_X^+$", fontsize=label_fontsize, labelpad=label_labelpad)
    
    plt.gca().invert_yaxis()
    
    dir_name = os.path.dirname(path)
    parts = dir_name.split('/')
    ckpt = parts[-2]
    smoothing_config = os.path.splitext(os.path.basename(path))[0]

    if joint:
        save_name = f'figs/{ckpt}-{smoothing_config}-{joint[0]}-{joint[1]}-{joint[2]}.png'
    else:
        save_name = f'figs/{ckpt}-{smoothing_config}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()    

if __name__ == "__main__":
    
    main()

