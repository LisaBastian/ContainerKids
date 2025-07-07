import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion(cm, title, out_path=None):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    if out_path:
        plt.savefig(out_path)
    plt.show()


def plot_uv_histogram(results):
    df = pd.DataFrame(results)
    df['UV_pair'] = df['U'].astype(str) + '-' + df['V'].astype(str)
    counts = df.groupby('UV_pair').size().reset_index(name='count')
    counts['UV_pair'] = pd.Categorical(
        counts['UV_pair'],
        categories=sorted(counts['UV_pair']),
        ordered=True
    )
    plt.figure(figsize=(10,6))
    sns.barplot(data=counts, x='UV_pair', y='count')
    plt.xticks(rotation=45)
    plt.xlabel("U-V Pair")
    plt.ylabel("Count")
    plt.title("Frequency of U-V Pairs")
    plt.tight_layout()
    plt.show()