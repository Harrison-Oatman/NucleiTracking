"""Run the pipeline's local_post steps on a seg3d config (seg3d_to_pipeline.py output) without
the napari step: merge_centroids (skips: new_centroids.csv exists), tracking, division mapping,
cylindrical coords, HDF5 export.

Usage: uv run python batched/run_seg3d_tracking.py <dataset>/tracking_<name>_seg3d/pipeline_config.yml
"""

import sys
import time
from pathlib import Path

from nucleitracking.pipeline.config import PipelineConfig
from nucleitracking.pipeline.steps import local_post


def main():
    cfg_path = Path(sys.argv[1])
    config = PipelineConfig.load(cfg_path)
    config.local_post.data_export.show_napari = False
    dataset = Path(config.dataset)
    print(f"dataset {dataset}, param set {config.param_set_name}", flush=True)
    for step in (
        local_post.run_merge_centroids,
        local_post.run_tracking,
        local_post.run_division_mapping,
        local_post.run_set_cylindrical_coords,
        local_post.run_export_hdf5,
    ):
        t = time.time()
        step(dataset, config)
        print(f"  {step.__name__}: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
