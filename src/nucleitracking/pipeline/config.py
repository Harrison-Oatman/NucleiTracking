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
    motion_model: str = "nearest_neighbor"


class DivisionMappingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    interphase_dividers: list[int] = Field(
        default_factory=lambda: [45, 80, 130, 195, 267]
    )
    new_track_cost: float = 25.0


class DataExportParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    anterior: float = -200
    posterior: float = 200
    dorsal_on_right: bool = True
    show_napari: bool = True


class LocalPostConfig(BaseModel):
    reconstruct_3d: Reconstruct3DParams = Field(default_factory=Reconstruct3DParams)
    tracking: TrackingParams = Field(default_factory=TrackingParams)
    division_mapping: DivisionMappingParams = Field(
        default_factory=DivisionMappingParams
    )
    data_export: DataExportParams = Field(default_factory=DataExportParams)


class Condition(Enum):
    WILD_TYPE = "wt"
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
                yaml.dump(data, f)
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
