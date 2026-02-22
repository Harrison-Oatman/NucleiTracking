# Nuclei Tracking Pipeline Documentation

This guide outlines how to use the automated `nucleitracking` pipeline. The workflow is designed to handle the complex process of segmenting and tracking nuclei in *Drosophila* blastoderm lightsheet data.

Because the full 3D data usually resides on a high-performance computing cluster, the pipeline is split into logical phases: Local Processing and Cluster Processing.

## Overview of Phases

1.  **Phase 1: Local Pre-processing** (`--phase local_pre`)
    *   Peak detection on the final timepoint
    *   Mesh generation
    *   UV Unwrapping
    *   **Action**: Transfer generated meshes to the cluster.
2.  **Phase 2: Cluster Execution** (`--phase cluster`)
    *   Project 3D volumes into 2D using UV unwrappings.
    *   Run Cellpose-SAM on the 2D projections.
    *   **Action**: Transfer 2D segmentation masks back to the local workstation.
3.  **Phase 3: Local Post-processing & Tracking** (`--phase local_post`)
    *   Reconstruct 2D segments back into 3D centroids.
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

cluster:
  project_2d: {}
  cellpose_sam:
    model_type: "cyto3"
    diameter: 15.0
    use_gpu: true

local_post:
  reconstruct_3d:
    max_distance: 5.0
  tracking:
    search_radius: 8.0
    max_gap_frames: 3
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

### 2. Run Cluster Processing

Log into your cluster. Ensure you have the same `config.yml` on the cluster, or simply use the automated copy saved by the runner inside the dataset folder.

Run the cluster logic (this requires GPU allocation):

```bash
python scripts/run_pipeline.py -c config.yml --phase cluster
```

**Data Transfer Step**:
Once the Cellpose segmentations are finished, the script will output another `rsync` command. Run this from your local workstation terminal to pull the segmentations back.

Example:
```bash
rsync -avz username@hpc.cluster.edu:/path/to/data/embryo1/tracking_test_run_001/segmented/ D:/Tracking/NucleiTracking/data/interim/lightsheet/embryo1/tracking_test_run_001/segmented/
```

### 3. Run Local Post-processing and Tracking

Back on your local workstation, run the final phase to reconstruct the 3D positions and track them:

```bash
python scripts/run_pipeline.py -c config.yml --phase local_post
```

This will run the built-in LAP tracker (bypassing TrackMate) and execute the mitosis mapping script. The final outputs will be saved in your dataset's `tracking_<param_set_name>` directory as `final_lineages.csv`.
