import pandas as pd
import matplotlib.pyplot as plt


csvList = ['./log_baseline.txt', './log_skip.txt', './log_size.txt', './log_skip_5.txt', './log_skip_size.txt', './log_skip_size_5.txt']

def generateGraphs():
    for csv in csvList:
        # Read CSV
        df = pd.read_csv(csv)

        # Plot latency
        plt.figure(figsize=(10, 5))
        plt.plot(df["Frame"], df["FPS"])
        plt.title("PFS per Frame")
        plt.xlabel("Frame")
        plt.ylabel("FPS")
        plt.grid(True)
        plt.ylim(0, 150)

        graphName = './imgs/fps_vs_frame_' + csv[6:-4]

        # Save graph
        plt.savefig(graphName, dpi=300, bbox_inches="tight")

        plt.show()

def calcAvgInfLatency():
    for csv in csvList:
        df = pd.read_csv(csv)
        last100 = df.tail(100)

        print(f'{csv}: {last100['Inference (ms)'].mean()}')


if __name__ == '__main__':
    calcAvgInfLatency()