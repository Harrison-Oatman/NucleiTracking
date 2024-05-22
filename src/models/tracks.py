import networkx as nx
import pandas as pd


class Spot:
    def __init__(self, idx):

        self.idx = idx
        self.parent = None
        self.children = []

    def add_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def __repr__(self):
        return f"Spot({self.idx})"


class Tracklet:
    def __init__(self, idx):

        self.idx = idx
        self.spots = {}
        self.parent = None
        self.children = []

    def add_spot(self, spot):
        self.spots[spot.idx] = spot

    def add_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def spots(self):
        return self.spots

    def __repr__(self):
        return f"Tracklet({self.idx})"


class Track:
    def __init__(self, idx):

        self.idx = idx
        self.graph = nx.DiGraph()
        self.tracklets = {}

    def add_tracklet(self, tracklet):
        self.graph.add_node(tracklet.idx, tracklet=tracklet)
        self.tracklets[tracklet.idx] = tracklet

    def add_edge(self, parent, child):
        self.graph.add_edge(parent.idx, child.idx)
        child.add_parent(parent)

    def spots(self):
        spots = dict()
        for tracklet in self.tracklets.values():
            spots.update(tracklet.spots)
        return spots


class Embryo:
    def __init__(self, name):

        self.name = name
        self.tracks = {}

    def add_track(self, track):
        self.tracks[track.idx] = track

    def spots(self):
        spots = dict()
        for track in self.tracks.values():
            spots.update(track.spots())
        return spots


def construct_from_df(df):
    embryo_names = df["root"].unique()

    return [Embryo(name) for name in embryo_names]
