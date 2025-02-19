from natsort import natsorted
import os
import pandas as pd
import json
from pathlib import Path


def load_peaks(directory):
    csv_files = natsorted([f for f in os.listdir(directory) if f.endswith('.csv')])
    peak_data = [pd.read_csv(os.path.join(directory, f)) for f in csv_files]
    stems = [Path(f).stem for f in csv_files]

    return peak_data, stems


class JsonController:

    def __init__(self, path):
        self.path = path
        self.json = load_json(path)

    def get(self, key, allkeys):

        nsortedjsonkeys = natsorted(self.json.keys())
        nsorted = natsorted(allkeys)

        idxs = [nsorted.index(k) for k in nsortedjsonkeys]
        idx = nsorted.index(key)

        # find first index in idxs that is less than idx

        for i, v in enumerate(idxs):
            if v > idx:
                break

        if i == 0:
            return self.json[nsortedjsonkeys[0]]
        else:
            return self.json[nsortedjsonkeys[i-1]]


def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)


def main():

    peak_directory = r"/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/recon/dog_sweep"
    json_fp = r"/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/recon/presets.json"

    peaks, stems = load_peaks(peak_directory)
    jc = JsonController(json_fp)

    dfs = []

    for i, s in enumerate(stems):
        preset = jc.get(s, stems)
        data = peaks[i]

        dog_preset = preset["dog"]
        min_distance_preset = preset["min-distance"]
        value = preset["value"]
        local_value = preset["local"]

        data = data[data["dog"] == dog_preset]
        data = data[data["min-distance"].astype(str) == min_distance_preset]
        data = data[data['val'] > value]
        data = data[data['local'] > local_value]

        dfs.append(data)

    df = pd.concat(dfs)
    df.to_csv("output.csv")
    

if __name__ == "__main__":
    main()