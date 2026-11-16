# Nuclei Tracking Pipeline Documentation

This guide outlines how to use the automated `nucleitracking` pipeline. The workflow is designed to handle the complex process of segmenting and tracking nuclei in *Drosophila* blastoderm lightsheet data.

Because the full 3D data usually resides on a high-performance computing cluster, the pipeline is split into logical phases: Local Processing and Cluster Processing.

## Overview of Phases

1.  **Phase 1: Local Pre-processing** (`--phase local_pre`)
    *   Peak detection on the final timepoint
    *   Mesh generation
    *   UV Unwrapping
    *   **Action**: Transfer generated meshes to the batch processing cluster.
2.  **Phase 2: Local Post-processing & Tracking** (`--phase local_post`)
    *   **Action**: Once 2D centroids are extracted on the cluster, bring `centroids_2d_from_stack.csv` back to the local workstation.
    *   Reconstruct 2D segments back into 3D centroids and merge across meshes.
    *   Run continuous frame-to-frame LAP tracking (replaces TrackMate).
    *   Map lineages through mitosis.

## Configuration File

The pipeline is driven entirely by a configuration file (YAML or JSON). This ensures all parameters are tracked and reproducible.

Here is a sample `config.yml`:

```yaml
dataset: "D:/Tracking/NucleiTracking/data/interim/lightsheet/embryo1"
param_set_name: "test_run_001"

local_pre:
  peak_detection:
    sigma_low: 2
    sigma_high: 6
    min_distance: 5
    threshold_abs: 35
  mesh_generation: {}
  uv_unwrap: {}


local_post:
  reconstruct_3d:
    max_distance: 5.0
  tracking:
    search_radius: 5.0
    max_gap_frames: 3
    start_frame: 26
    skip_frames: [134, 135]
    motion_model: "nearest_neighbor"
  division_mapping:
    interphase_dividers: [45, 80, 130, 195, 267]
    new_track_cost: 25.0
```

`tracking.motion_model` selects the frame-to-frame linker:

- `nearest_neighbor` (default): LAP on raw positions (`models/lap_tracking.py`).
- `velocity`: LAP on predicted positions (`models/motion_tracking.py`). Each
  nucleus is projected forward by a blend of its own smoothed velocity and the
  median velocity of its neighbours (`velocity_weight`, default 0.5), scaled by
  the real frame interval from the `json/` timestamps, and each frame pair is
  re-solved with the flow measured from the first pass. Use it when nuclei move
  a large fraction of their spacing per frame (the post-NC13 spreading) or the
  acquisition has pauses. On 20250625 it cut track breaks in the NC13→14
  window from 494 to ~200 and links with >6 px acceleration from 3464 to ~390,
  while matching nearest-neighbour tracking on 20250131.

`division_mapping` links daughters to mothers in each mitotic wave:

```yaml
  division_mapping:
    mitosis_windows: [[61, 69], [105, 115], [155, 170], [238, 257]]  # raw frames
    method: hypotheses
    new_track_cost: 36.0   # squared px; 6 px max midpoint error
```

- `mitosis_windows` gives the `[start, end]` frames of each wave (pad the burst
  of new track starts by ~2 frames). Daughters may only start inside a window
  and mothers must exist before it, so a track that breaks and restarts in
  interphase is no longer mapped as a daughter. Cycle numbers in the export are
  counted from the window starts. Without it, windows are derived from
  `interphase_dividers` as before.
- `method: lap` is the original one-LAP-per-window assignment.
  `method: hypotheses` keeps the backward-interpolated midpoint cost but
  enumerates explicit hypotheses and picks the best non-conflicting set with an
  integer program (one division per mother per window):
  - one new daughter: the mother continues into one daughter, a new track is
    the other;
  - two new daughters: the mother's track ends inside the window and two new
    tracks appear within `max_mother_gap` frames;
  - mothers are measured at their last *real* detection, up to
    `max_mother_gap` frames before the daughter, so a nucleus missed in
    anaphase (bridged by gap closing and interpolation) is not measured at a
    fake interpolated position;
  - any track established before the window can be a mother, and any track
    starting in it with `min_daughter_len` real spots can be a daughter;
  - `allow_cut: true` also lets a mother drop a wrong continuation in favour
    of two new daughters (rarely fires; off by default).

To compare tracking variants, give each its own `param_set_name`; outputs go to
`tracking_<param_set_name>/`. Copying an existing `new_centroids.csv` into the
new folder skips the (interactive) centroid merge.

## Running the Pipeline

You will use the `run_pipeline.py` script located in `scripts/`.

### 1. Run Local Pre-processing

On your workstation, run:

```bash
python scripts/run_pipeline.py -c config.yml --phase local_pre
```

**Data Transfer Step**:
When phase 1 is complete, the script will output an `rsync` command. You must run this command to transfer your newly generated `.obj` meshes and UV unwrappings to the cluster.

Example:
```bash
rsync -avz D:/Tracking/NucleiTracking/data/interim/lightsheet/embryo1/tracking_test_run_001/uv_unwrap/ username@hpc.cluster.edu:/path/to/data/embryo1/tracking_test_run_001/uv_unwrap/
```

### 2. Run Local Post-processing and Tracking

Once the batch scripts on the cluster have finished segmenting and finding the 2D properties, ensure you transfer `centroids_2d_from_stack.csv` back into the local tracking directory.

Back on your local workstation, run the final phase to reconstruct the 3D positions and track them:

```bash
python scripts/run_pipeline.py -c config.yml --phase local_post
```

This will run the built-in tracking merging algorithm, the LAP tracker (bypassing TrackMate), and execute the mitosis mapping script. The final outputs will be saved in your dataset's `tracking_<param_set_name>` directory as `final_lineages.csv`.
