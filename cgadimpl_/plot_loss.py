
import matplotlib.pyplot as plt
import csv

steps = []
losses = []

try:
    with open('/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/loss.txt', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            steps.append(int(row[0]))
            losses.append(float(row[1]))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss (Cross Entropy)')
    plt.title('Word Prediction Training Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig('/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/loss_plot_bigram.png')
    print("Graph saved to loss_plot_bigram.png")

except Exception as e:
    print(f"Error: {e}")
