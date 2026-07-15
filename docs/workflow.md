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
