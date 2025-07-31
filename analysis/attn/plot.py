import json
import matplotlib.pyplot as plt
import numpy as np


for device in ['mi300x', 'mi325x', 'mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x - width, aiter_tflops, width, label='AITER (AMD)', alpha=0.8)
    bars1 = ax.bar(x, tk_tflops, width, label='ThunderKittens', alpha=0.8)

    max_tflops = max(max(aiter_tflops), max(tk_tflops))

    # Add value labels on bars
    for bar, value in zip(bars0, aiter_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=9)

    for bar, value in zip(bars1, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Sequence Length (N)')
    ax.set_ylabel('Performance (TFLOPS)')
    ax.set_title(f'Flash Attention Performance Comparison {device.upper()}')
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_attn_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
    print(f"TK TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

