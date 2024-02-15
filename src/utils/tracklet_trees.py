from ete3 import Tree
import matplotlib.pyplot as plt
from math import floor


def get_tracklet_tree(tracklets):

    sorted_tracklets = tracklets.sort_values(by="start_time")

    tracklet_tree = Tree()

    tracklet_tree.add_child(name=str(int(sorted_tracklets.index[0])))
    tracklet_tree.children[-1].add_features(start=sorted_tracklets["start_time"].iloc[0], end=sorted_tracklets["end_time"].iloc[0])

    for idx, row in sorted_tracklets.iterrows():
        idx = str(int(idx))
        parent_idx = row["parent"]
        if parent_idx == 0:
            continue
        else:
            parent = tracklet_tree.search_nodes(name=str(int(parent_idx)))[0]
            parent.add_child(name=idx, dist=row["length"])
            parent.children[-1].add_features(start=row["start_time"], end=row["end_time"])

    return tracklet_tree


def visualize_tracklet_tree(tracklet_tree: Tree, ax=None, c="black"):
    for i, leaf in enumerate(tracklet_tree.iter_leaves()):
        leaf.name = str(leaf.name)
        leaf.add_features(y=i)

    for node in tracklet_tree.traverse(strategy="postorder"):
        if node.children:
            node.add_feature("y", node.children[0].y + (node.children[-1].y - node.children[0].y) / 2)

    # print(tracklet_tree.get_ascii(attributes=["name", "y", "start", "end"]))

    if ax is None:
        fig, ax = plt.subplots()

    for node in tracklet_tree.traverse():

        # skip root
        if node.is_root():
            continue

        # draw branch
        x = [node.start, node.end]
        y = [node.y, node.y]
        ax.plot(x, y, color=c, lw=2)

        # label branch length
        x = node.end - node.dist / 2
        y = node.y
        ax.text(x, y + 0.1, f"{floor(node.dist // 1)}:{floor((node.dist % 1)*60):02d}", ha="center", va="bottom")

        if node.children:
            x = [node.end for child in node.children]
            y = [child.y for child in node.children]
            ax.plot(x, y, color=c, lw=2)

        # if node.is_leaf():
        #     ax.text(node.end + 0.5, node.y, node.name, ha="left", va="center")

    ax.yaxis.set_visible(False)
#
#
# def main():
#     pass
#
# if __name__ == "__main__":
#     main()