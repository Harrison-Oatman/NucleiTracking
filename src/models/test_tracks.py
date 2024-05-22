import unittest
from .tracks import Spot, Tracklet, Track, Embryo


def construct_embryo():
    embryo = Embryo("embryo")
    track = Track(0)
    tracklet = Tracklet(0)
    spot = Spot(0)
    tracklet.add_spot(spot)
    track.add_tracklet(tracklet)
    embryo.add_track(track)
    return embryo


class TestEmbryo(unittest.TestCase):

    def test_add_track_returns_dict(self):
        embryo = Embryo("embryo")
        track = Track(0)
        embryo.add_track(track)
        self.assertEqual(embryo.tracks, {0: track})

    def test_return_spots(self):
        embryo = construct_embryo()
        self.assertEqual(embryo.spots(), {0: embryo.tracks[0].tracklets[0].spots[0]})

    def test_return_spots_empty(self):
        embryo = Embryo("embryo")
        self.assertEqual(embryo.spots(), {})


if __name__ == '__main__':
    unittest.main()
