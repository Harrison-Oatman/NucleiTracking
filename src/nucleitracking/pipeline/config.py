import json
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PeakDetectionParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    sigma_low: float = 2.0
    sigma_high: float = 6.0
    min_distance: int = 5
    threshold_abs: float = 35.0


class MeshGenerationParams(BaseModel):
    model_config = ConfigDict(extra="allow")


class UVUnwrapParams(BaseModel):
    model_config = ConfigDict(extra="allow")


class LocalPreConfig(BaseModel):
    peak_detection: PeakDetectionParams = Field(default_factory=PeakDetectionParams)
    mesh_generation: MeshGenerationParams = Field(default_factory=MeshGenerationParams)
    uv_unwrap: UVUnwrapParams = Field(default_factory=UVUnwrapParams)


class Reconstruct3DParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_distance: float = 5.0


class TrackingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    search_radius: float = 5.0
    max_gap_frames: int = 3
    start_frame: int = 0
    skip_frames: list[int] = Field(default_factory=list)
    # "nearest_neighbor": LAP on raw positions (models/lap_tracking.py)
    # "velocity": LAP on positions predicted from track velocity and local flow
    #             (models/motion_tracking.py); better when nuclei move a large
    #             fraction of their spacing per frame, e.g. after NC13 mitosis
    motion_model: str = "nearest_neighbor"
    velocity_weight: float = 0.5  # "velocity" only: own velocity vs. local flow


class DivisionMappingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    # [start, end] frames (inclusive) of each mitotic wave. Daughters may only
    # start inside a window and mothers must exist before it. If omitted, the
    # windows are derived from interphase_dividers: (d_k, d_k+1] for each pair.
    mitosis_windows: list[tuple[int, int]] | None = None
    interphase_dividers: list[int] = Field(
        default_factory=lambda: [45, 80, 130, 195, 267]
    )
    # "lap": one LAP per window, new tracklet -> full-length tracklet
    # "hypotheses": per-window integer program over one- and two-new-daughter
    #               hypotheses (models/division_tracking.py)
    method: str = "lap"
    # squared px: the LAP's no-assignment cost / the hypotheses' maximum
    # midpoint error (36 = 6 px was best on 20250131 and the paper embryos)
    new_track_cost: float = 25.0
    max_mother_gap: int = 3  # "hypotheses": frames between mother and daughter
    min_daughter_len: int = 3  # "hypotheses": real spots in a daughter tracklet
    allow_cut: bool = False  # "hypotheses": may cut a wrong mother continuation

    def windows(self) -> list[tuple[int, int]]:
        if self.mitosis_windows:
            return [(int(s), int(e)) for s, e in self.mitosis_windows]
        d = self.interphase_dividers
        return [(d[k] + 1, d[k + 1]) for k in range(len(d) - 1)]

    def cycle_boundaries(self) -> list[int]:
        """First frame of each new cycle (a tracklet starting here is a daughter)."""
        if self.mitosis_windows:
            return [int(s) for s, _ in self.mitosis_windows]
        return list(self.interphase_dividers)


class CylidricalCoordsParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    anterior: float = -200
    posterior: float = 200
    dorsal_on_right: bool = True
    show_napari: bool = True


class Merge3DTrackingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    csv_path: str | None = None  # defaults to tracking_dir/centroids_3d.csv
    crop_shape: list[int] | None = None  # [z_size, y_size, x_size] for edge rejection
    start_frame: int | None = None  # only use 3D centroids at frames >= this
    end_frame: int | None = None  # only use 3D centroids at frames <= this
    edge_cutoff: float = 10.0  # reject centroids within this many voxels of crop edge
    max_merge_distance: float = (
        25.0  # spatial distance to consider centroid a duplicate
    )


class LocalPostConfig(BaseModel):
    reconstruct_3d: Reconstruct3DParams = Field(default_factory=Reconstruct3DParams)
    merge_3d_tracking: Merge3DTrackingParams = Field(
        default_factory=Merge3DTrackingParams
    )
    tracking: TrackingParams = Field(default_factory=TrackingParams)
    division_mapping: DivisionMappingParams = Field(
        default_factory=DivisionMappingParams
    )
    data_export: CylidricalCoordsParams = Field(default_factory=CylidricalCoordsParams)


class Condition(Enum):
    WILD_TYPE = "wt"
    WIlD_TYPE_WEIRD = "wt*"
    BCD = "bcd"
    TRK = "trk"


class MetadataConfig(BaseModel):
    um_per_px: float = 0.58275
    condition: Condition = Condition.WILD_TYPE


class PipelineConfig(BaseModel):
    dataset: Path
    param_set_name: str = "default"

    local_pre: LocalPreConfig = Field(default_factory=LocalPreConfig)
    local_post: LocalPostConfig = Field(default_factory=LocalPostConfig)
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)

    @classmethod
    def load(cls, path: Path | str) -> "PipelineConfig":
        path = Path(path)
        if path.suffix in [".yml", ".yaml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            with open(path) as f:
                data = json.load(f)
        return cls(**data)

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        if path.suffix in [".yml", ".yaml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
