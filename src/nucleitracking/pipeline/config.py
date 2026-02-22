import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class PeakDetectionParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Add relevant parameters here later based on research


class MeshGenerationParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    # e.g., resolution, smoothing


class UVUnwrapParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    # e.g., overlapping margins


class LocalPreConfig(BaseModel):
    peak_detection: PeakDetectionParams = Field(default_factory=PeakDetectionParams)
    mesh_generation: MeshGenerationParams = Field(default_factory=MeshGenerationParams)
    uv_unwrap: UVUnwrapParams = Field(default_factory=UVUnwrapParams)


class Project2DParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    # e.g., depth of max projection


class CellposeSAMParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    model_type: str = "cyto3"
    diameter: float = 15.0
    use_gpu: bool = True


class ClusterConfig(BaseModel):
    project_2d: Project2DParams = Field(default_factory=Project2DParams)
    cellpose_sam: CellposeSAMParams = Field(default_factory=CellposeSAMParams)


class Reconstruct3DParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    max_distance: float = 5.0


class TrackingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    search_radius: float = 5.0
    max_gap_frames: int = 3
    motion_model: str = "nearest_neighbor"


class DivisionMappingParams(BaseModel):
    model_config = ConfigDict(extra="allow")
    interphase_dividers: list[int] = Field(
        default_factory=lambda: [45, 80, 130, 195, 267]
    )
    new_track_cost: float = 25.0


class LocalPostConfig(BaseModel):
    reconstruct_3d: Reconstruct3DParams = Field(default_factory=Reconstruct3DParams)
    tracking: TrackingParams = Field(default_factory=TrackingParams)
    division_mapping: DivisionMappingParams = Field(
        default_factory=DivisionMappingParams
    )


class PipelineConfig(BaseModel):
    dataset: Path
    param_set_name: str = "default"

    local_pre: LocalPreConfig = Field(default_factory=LocalPreConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    local_post: LocalPostConfig = Field(default_factory=LocalPostConfig)

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
