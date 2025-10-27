import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def plotset(imgs, shape, reshape=None, cmap='gray_r'):
    fig, ax_array = plt.subplots(shape, shape)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        img = imgs[i]
        if reshape is not None:
            img = img.reshape(reshape)
        ax.imshow(img, cmap=cmap)
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.01)
    plt.show()


def order_clustering(children):
    """
    This function takes the children of a clustering and returns the order of the leaves
    """

    n = len(children) + 1

    def recursive_split(k):
        if k < n:
            return [k]
        left, right = children[k - n]
        return recursive_split(left) + recursive_split(right)

    return recursive_split(len(children) + n - 1)


def plot_transition_heatmap(nucleus_df, cluster_kword="kmeans_cluster"):
    nucleus_df["cluster_cat"] = nucleus_df[cluster_kword].astype("int").astype("category")
    parent_cluster = [nucleus_df["cluster_cat"][p] if not pd.isna(p) else None for p in nucleus_df["parent"]]
    nucleus_df["parent_cluster_cat"] = parent_cluster

    transitions = np.array(pd.crosstab(nucleus_df["cluster_cat"], nucleus_df["parent_cluster_cat"]))
    distance = 1 / (np.array(transitions + 1))
    transition_probs = transitions / np.sum(transitions, axis=0)

    clustering = AgglomerativeClustering(metric="precomputed", linkage="single")
    clustering.fit(distance)

    order = order_clustering(clustering.children_)
    order = order[::-1]

    transitions_reorder = transition_probs[order, :]
    transitions_reorder = transitions_reorder[:, order]

    sns.heatmap(np.log2(transitions_reorder + 1 / 2 ** 8), annot=True, xticklabels=order, yticklabels=order)
    plt.xlabel("Parent Cluster")
    plt.ylabel("Child Cluster")
    plt.title("Transition Probabilities")
    plt.show()
