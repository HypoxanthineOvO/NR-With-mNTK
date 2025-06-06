import numpy as np
import torch
import matplotlib.pyplot as plt
import os, sys, shutil
from NTK import GetEigenValuesData

def parse_NTK_in_training(ntks: dict) -> dict:
    results = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_encoding": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_grids": [
            [] for _ in range(16)  # Assuming 16 hash grids
        ],
        "psnr": None
    }
    key: str
    value: torch.Tensor
    for ntk in ntks:
        for key, value in ntks[0].items():
            if key.startswith("hash_group_"):
                group_i = key.split("_")[-1]
                assert group_i.isdigit(), f"Key '{key}' does not end with a valid group index."
                group_id = int(group_i)
                eigval = GetEigenValuesData(value)
                results["hash_grids"][group_id].append(
                    eigval['max']
                )
                continue

            eigvals = GetEigenValuesData(value)
            #print(f"Max Eigenvalue: {eigvals['max']:.4e}, Condition Number: {eigvals['cond_num']:.4e}")
            if key == "hash_network":
                results["hash_network"]["Max Eigenvalue"].append(eigvals["max"])
                results["hash_network"]["Condition Number"].append(eigvals["cond_num"])
            elif key == "rgb_network":
                results["rgb_network"]["Max Eigenvalue"].append(eigvals["max"])
                results["rgb_network"]["Condition Number"].append(eigvals["cond_num"])
            elif key == "hash_encoding":
                results["hash_encoding"]["Max Eigenvalue"].append(eigvals["max"])
                results["hash_encoding"]["Condition Number"].append(eigvals["cond_num"])
            
        
    results["hash_network"]["Max Eigenvalue"] = np.array(results["hash_network"]["Max Eigenvalue"]).mean()
    results["hash_network"]["Condition Number"] = np.array(results["hash_network"]["Condition Number"]).mean()
    results["rgb_network"]["Max Eigenvalue"] = np.array(results["rgb_network"]["Max Eigenvalue"]).mean()
    results["rgb_network"]["Condition Number"] = np.array(results["rgb_network"]["Condition Number"]).mean()
    results["hash_encoding"]["Max Eigenvalue"] = np.array(results["hash_encoding"]["Max Eigenvalue"]).mean()
    results["hash_encoding"]["Condition Number"] = np.array(results["hash_encoding"]["Condition Number"]).mean()
    
    for group_id, eigvals in enumerate(results["hash_grids"]):
        results["hash_grids"][group_id] = np.array(eigvals).mean()
        print(f"Hash Grid {group_id} - Max Eigenvalue: {results['hash_grids'][group_id]:.4e}")
    

    print("Lambda Max:")
    print(f"Hash Encoding: {results['hash_encoding']['Max Eigenvalue']:.4e}")
    print(f"Hash Network: {results['hash_network']['Max Eigenvalue']:.4e}") 
    print(f"RGB Network: {results['rgb_network']['Max Eigenvalue']:.4e}")
    return results


def parse_NTK(ntks: dict) -> dict:
    results = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_encoding": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_grids": [
            [] for _ in range(16)  # Assuming 16 hash grids
        ],
        "psnr": ntks["psnr"] if "psnr" in ntks else None
    }
    key: str
    value: torch.Tensor
    for key, value in ntks.items():
        if key == "hash_grids":
            for group_id in range(16):
                results["hash_grids"][group_id].append(
                    value[group_id].item()
                )
            continue
        
        #print(f"Max Eigenvalue: {eigvals['max']:.4e}, Condition Number: {eigvals['cond_num']:.4e}")
        if key == "hash_network":
            results["hash_network"]["Max Eigenvalue"].append(value["Max Eigenvalue"].item())
            results["hash_network"]["Condition Number"].append(value["Condition Number"].item())
        elif key == "rgb_network":
            results["rgb_network"]["Max Eigenvalue"].append(value["Max Eigenvalue"].item())
            results["rgb_network"]["Condition Number"].append(value["Condition Number"].item())
        elif key == "hash_encoding":
            results["hash_encoding"]["Max Eigenvalue"].append(value["Max Eigenvalue"].item())
            results["hash_encoding"]["Condition Number"].append(value["Condition Number"].item())
    
    results["hash_network"]["Max Eigenvalue"] = np.array(results["hash_network"]["Max Eigenvalue"]).mean()
    results["hash_network"]["Condition Number"] = np.array(results["hash_network"]["Condition Number"]).mean()
    results["rgb_network"]["Max Eigenvalue"] = np.array(results["rgb_network"]["Max Eigenvalue"]).mean()
    results["rgb_network"]["Condition Number"] = np.array(results["rgb_network"]["Condition Number"]).mean()
    results["hash_encoding"]["Max Eigenvalue"] = np.array(results["hash_encoding"]["Max Eigenvalue"]).mean()
    results["hash_encoding"]["Condition Number"] = np.array(results["hash_encoding"]["Condition Number"]).mean()
    #results["psnr"] = results["psnr"].item() if isinstance(results["psnr"], torch.Tensor) else results["psnr"]
    
    for group_id, eigvals in enumerate(results["hash_grids"]):
        results["hash_grids"][group_id] = np.array(eigvals).mean()
    return results

def Load_NTK(path: str) -> torch.Tensor:
    ntks = torch.load(path, map_location=torch.device('cpu'), weights_only=False)

    results = parse_NTK(ntks)
    # print(f"Hash Network - Max Eigenvalue: {results['hash_network']['Max Eigenvalue']:.4e}, "
    #       f"Condition Number: {results['hash_network']['Condition Number']:.4e}")
    # print(f"RGB Network - Max Eigenvalue: {results['rgb_network']['Max Eigenvalue']:.4e}, "
    #       f"Condition Number: {results['rgb_network']['Condition Number']:.4e}")
    # print(f"Hash Encoding - Max Eigenvalue: {results['hash_encoding']['Max Eigenvalue']:.4e}, "
    #         f"Condition Number: {results['hash_encoding']['Condition Number']:.4e}")
    return results

if __name__ == "__main__":
    base_path: str = "./Results/Baseline"
    result_to_plot = {
        "hash_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "rgb_network": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_encoding": {
            "Max Eigenvalue": [],
            "Condition Number": [],
        },
        "hash_grids": [[] for _ in range(16)],  # Assuming 16 hash grids
        "psnrs": []
    }
    max_steps = 1000 * 25
    for step in range(0, max_steps, 1000):
        print(f"Loading NTK data for step {step}...")
        print("=" * 50)
        result = Load_NTK(os.path.join(base_path, f"lego_ntk_{step}.pt"))
        result_to_plot["hash_network"]["Max Eigenvalue"].append(result["hash_network"]["Max Eigenvalue"])
        result_to_plot["hash_network"]["Condition Number"].append(result["hash_network"]["Condition Number"])
        result_to_plot["rgb_network"]["Max Eigenvalue"].append(result["rgb_network"]["Max Eigenvalue"])
        result_to_plot["rgb_network"]["Condition Number"].append(result["rgb_network"]["Condition Number"])
        result_to_plot["hash_encoding"]["Max Eigenvalue"].append(result["hash_encoding"]["Max Eigenvalue"])
        result_to_plot["hash_encoding"]["Condition Number"].append(result["hash_encoding"]["Condition Number"])
        result_to_plot["psnrs"].append(result["psnr"] if result["psnr"] is not None else 0.0)
        for group_id in range(16):
            result_to_plot["hash_grids"][group_id].append(result["hash_grids"][group_id])


    hash_max_eigenvalues = result_to_plot["hash_network"]["Max Eigenvalue"]
    rgb_max_eigenvalues = result_to_plot["rgb_network"]["Max Eigenvalue"]

    max_val = max(max(hash_max_eigenvalues), max(rgb_max_eigenvalues))

    plt.figure(figsize=(22, 6), dpi = 150)
    # Font: Arial
    plt.rcParams['font.family'] = 'serif'
    plt.subplot(1, 3, 1)
    


    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["hash_network"]["Max Eigenvalue"], 
        color = "#EECA40",
        label='Hash Network Max Eigenvalue', marker='o'
    )
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["rgb_network"]["Max Eigenvalue"],
        color = "#FD763F",
        label="RGB Network Max Eigenvalue", marker='o'
    )
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["hash_encoding"]["Max Eigenvalue"],
        color = "#23BAC5",
        label="Hash Encoding Max Eigenvalue", marker='o'
    )
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Max Eigenvalue', fontsize=14)
    # Logarithmic scale for better visibility
    plt.yscale('log')
    plt.title('Max Eigenvalue of Networks Over Training Steps', fontsize=18)
    plt.legend()
    plt.grid()
    plt.subplot(1, 3, 2)

    # Use a qualitative colormap with good distinction (e.g., 'tab20' for 16 groups)
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in np.linspace(0, 1, 16)]

    for group_id in range(16):
        plt.plot(
            range(0, max_steps, 1000), 
            result_to_plot["hash_grids"][group_id], 
            color=colors[group_id],
            linestyle='-',
            linewidth=1.5,
            marker='o',
            markersize=4,
            label=f'Grid {group_id}'
        )

    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Max Eigenvalue', fontsize=14)
    plt.title('Max Eigenvalue of Hash Grids Over Training Steps', 
            fontsize=18)

    # Improve legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(
        handles, labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        framealpha=1,
        title='Grid ID',
        title_fontsize=10
    )

    # Grid and layout improvements
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.plot(
        range(0, max_steps, 1000), 
        result_to_plot["psnrs"],
        label='PSNR (scaled)', marker='o'
    )
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('PSNR', fontsize=14)
    plt.title('PSNR Over Training Steps', fontsize=18)
    plt.yscale('linear')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('ngp_eigenvalue_plot.png')