from pathlib import Path
from typing import Literal

from nucleitracking.pipeline import steps
from nucleitracking.pipeline.config import PipelineConfig


class PipelineRunner:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.config = PipelineConfig.load(self.config_path)

    def run(self, phase: Literal["local_pre", "cluster", "local_post", "all"]):
        print(f"Starting NucleiTracking pipeline - Phase: {phase.upper()}")
        print(f"Processing dataset: {self.config.dataset.name}")

        dataset = Path(self.config.dataset)
        if not dataset.exists():
            print(f"Error: Dataset path does not exist: {dataset}")
            return

        # Ensure tracking directory exists inside the dataset to save config/outputs
        track_dir = dataset / f"tracking_{self.config.param_set_name}"
        track_dir.mkdir(parents=True, exist_ok=True)

        # Save a copy of the config for reproducibility
        self.config.save(track_dir / "pipeline_config.yml")

        if phase in ["local_pre", "all"]:
            self._run_local_pre(dataset)

        if phase in ["cluster", "all"]:
            self._run_cluster(dataset)

        if phase in ["local_post", "all"]:
            self._run_local_post(dataset)

        # Print data transfer instructions at the boundaries
        if phase == "local_pre":
            print("\n" + "=" * 50)
            print("LOCAL PRE-PROCESSING COMPLETE.")
            print("ACTION REQUIRED: Transer meshes and UVs to cluster.")
            print("Example rsync command:")
            track_dir = (
                Path(self.config.dataset) / f"tracking_{self.config.param_set_name}"
            )
            print(
                f"  rsync -avz {track_dir}/uv_unwrap/ user@cluster:/path/to/cluster/{self.config.dataset.name}/uv_unwrap/"
            )
            print("=" * 50 + "\n")

        elif phase == "cluster":
            print("\n" + "=" * 50)
            print("CLUSTER PROCESSING COMPLETE.")
            print("ACTION REQUIRED: Transer masks and projections back to local.")
            print("Example rsync command:")
            track_dir = (
                Path(self.config.dataset) / f"tracking_{self.config.param_set_name}"
            )
            print(
                f"  rsync -avz user@cluster:/path/to/cluster/{self.config.dataset.name}/segmented/ {track_dir}/segmented/"
            )
            print("=" * 50 + "\n")

        elif phase in ["local_post", "all"]:
            print("\n" + "=" * 50)
            print("PIPELINE COMPLETE.")
            print("=" * 50 + "\n")

    def _run_local_pre(self, dataset: Path):
        print(f"  [{dataset.name}] --- local_pre ---")
        steps.run_peak_detection(dataset, self.config)
        steps.run_mesh_generation(dataset, self.config)
        steps.run_uv_unwrapping(dataset, self.config)

    def _run_cluster(self, dataset: Path):
        print(f"  [{dataset.name}] --- cluster ---")
        steps.run_project_2d(dataset, self.config)
        steps.run_cellpose_sam(dataset, self.config)

    def _run_local_post(self, dataset: Path):
        print(f"  [{dataset.name}] --- local_post ---")
        steps.run_reconstruct_3d(dataset, self.config)
        steps.run_tracking(dataset, self.config)
        steps.run_division_mapping(dataset, self.config)
