# FedSim/plot_analysis.py

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# --- Paste Your Experimental Data Here ---

# Data from your FedAvg server logs and client reports
fedavg_data = {
    "global_test_accuracy": [99.55, 98.88, 95.13, 92.82, 91.41],
    "client_post_train_local_accuracy": {
        # Average of all clients' post-train accuracy (which was 100% for all)
        "round_1": 100.0,
        "round_2": 100.0,
        "round_3": 100.0,
        "round_4": 100.0,
        "round_5": 100.0,
    }
}

# Data from your FedProx simulation logs
fedprox_data = {
    "global_test_accuracy": [99.42, 98.72, 96.75, 96.85, 96.13],
    "client_post_train_local_accuracy": {
        "round_1": np.mean([84.40, 100.00, 100.00, 100.00]),
        "round_2": np.mean([100.00, 87.25, 100.00, 100.00]),
        "round_3": np.mean([100.00, 100.00, 90.16, 100.00]),
        "round_4": np.mean([100.00, 100.00, 91.44, 100.00]),
        "round_5": np.mean([100.00, 100.00, 100.00, 92.64]),
    }
}

# --- Plot style configuration (tweak these numbers to taste) ---
PLOT_STYLE = {
    "figsize": (12, 7),
    "title_size": 20,
    "label_size": 16,
    "tick_size": 14,
    "legend_size": 14,
    "text_size": 14,
    "line_width": 2.2,
    "marker_size": 9,
    "dpi": 150
}

# Apply rcParams for consistent font sizing
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "figure.figsize": PLOT_STYLE["figsize"],
    "axes.titlesize": PLOT_STYLE["title_size"],
    "axes.labelsize": PLOT_STYLE["label_size"],
    "xtick.labelsize": PLOT_STYLE["tick_size"],
    "ytick.labelsize": PLOT_STYLE["tick_size"],
    "legend.fontsize": PLOT_STYLE["legend_size"],
    "font.size": PLOT_STYLE["text_size"],
    "lines.linewidth": PLOT_STYLE["line_width"],
})

def plot_global_performance_comparison(fedavg_acc, fedprox_acc):
    """
    Plots the most important comparison: the final aggregated model's accuracy
    on the definitive test set over the rounds.
    """
    print("Generating Plot 1: Global Model Performance Comparison...")
    rounds = np.arange(1, len(fedavg_acc) + 1)

    plt.figure()  # uses figsize from rcParams

    plt.plot(rounds, fedavg_acc, marker='o', linestyle='--',
             label='FedAvg Global Accuracy', markersize=PLOT_STYLE["marker_size"])
    plt.plot(rounds, fedprox_acc, marker='s', linestyle='-',
             label='FedProx Global Accuracy (mu=0.01)', markersize=PLOT_STYLE["marker_size"])

    plt.title('FedAvg vs. FedProx: Global Model Performance on Test Set')
    plt.xlabel('Federated Round')
    plt.ylabel('Test Set Accuracy (%)')
    plt.xticks(rounds)
    plt.ylim(min(min(fedavg_acc), min(fedprox_acc)) - 2, 101)  # Dynamic y-axis
    plt.legend()
    plt.grid(True)

    # Add text for final accuracy (bigger font)
    plt.text(rounds[-1], fedavg_acc[-1], f'{fedavg_acc[-1]:.2f}%',
             ha='right', va='top', fontsize=PLOT_STYLE["text_size"], color='blue')
    plt.text(rounds[-1], fedprox_acc[-1], f'{fedprox_acc[-1]:.2f}%',
             ha='right', va='bottom', fontsize=PLOT_STYLE["text_size"], color='green')

    output_path = Path(__file__).parent / 'global_performance_comparison.png'
    plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    print(f"  > Plot saved to {output_path}")
    plt.show()


def plot_client_overfitting_comparison(fedavg_local_acc, fedprox_local_acc):
    """
    Plots the indicator of client overfitting: the average accuracy clients
    achieve on their own data after local training.
    """
    print("\nGenerating Plot 2: Client-Side Overfitting Comparison...")
    rounds = np.arange(1, len(fedavg_local_acc) + 1)

    plt.figure()

    plt.plot(rounds, fedavg_local_acc, marker='o', linestyle='--',
             label='FedAvg (Overfitting)', markersize=PLOT_STYLE["marker_size"])
    plt.plot(rounds, fedprox_local_acc, marker='s', linestyle='-',
             label='FedProx (Regularized)', markersize=PLOT_STYLE["marker_size"])

    plt.title('Client-Side Overfitting: Average Local Accuracy After Training')
    plt.xlabel('Federated Round')
    plt.ylabel('Average Post-Train Local Accuracy (%)')
    plt.xticks(rounds)
    plt.ylim(80, 101)
    plt.legend()
    plt.grid(True)

    output_path = Path(__file__).parent / 'client_overfitting_comparison.png'
    plt.savefig(output_path, dpi=PLOT_STYLE["dpi"], bbox_inches='tight')
    print(f"  > Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    # Extract data for plotting
    fedavg_global = fedavg_data["global_test_accuracy"]
    fedprox_global = fedprox_data["global_test_accuracy"]

    fedavg_local = list(fedavg_data["client_post_train_local_accuracy"].values())
    fedprox_local = list(fedprox_data["client_post_train_local_accuracy"].values())

    # Generate the plots
    plot_global_performance_comparison(fedavg_global, fedprox_global)
    plot_client_overfitting_comparison(fedavg_local, fedprox_local)
