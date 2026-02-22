from pathlib import Path

import natsort
import numpy as np
import tifffile
from blender_tissue_cartography import diffgeo
from blender_tissue_cartography import interpolation as tcinterp
from blender_tissue_cartography import mesh as tcmesh
from tqdm import tqdm

from nucleitracking.pipeline.config import PipelineConfig


def run_project_2d(dataset: Path, config: PipelineConfig):
    """
    Projects the 3D volumetric tif sequence onto the 2D UV unwrappings.
    Heavily adapted from batched/uv_unwrap.py.
    """
    print(f"[{dataset.name}] Running 2D Projection (Cluster)...")

    # Find the input tifs
    recon_dir = dataset / "recon"
    if not recon_dir.exists():
        print(f"  [Error] recon directory not found at {recon_dir}")
        return

    tif_files = natsort.natsorted(
        [f for f in recon_dir.iterdir() if f.suffix == ".tif"]
    )
    if not tif_files:
        print(f"  [Error] No tif files in {recon_dir}")
        return

    out_dir = dataset / f"tracking_{config.param_set_name}"
    uv_unwrap_dir = out_dir / "uv_unwrap"

    if not uv_unwrap_dir.exists() or not list(uv_unwrap_dir.glob("*.obj")):
        print(
            f"  [Error] No UV .obj files found in {uv_unwrap_dir}. Please run local_pre phase and transfer data."
        )
        return

    # Read bounds/resolution parameters
    # Hardcoded defaults currently reflecting old uv_unwrap.py
    normal_offsets = np.linspace(-12, 8, 21)
    uv_grid_steps = 1024
    resolution = (1, 1, 1)

    # We will process each obj file
    for obj_fp in uv_unwrap_dir.glob("*.obj"):
        obj_name = obj_fp.stem
        print(f"  Processing UV mesh: {obj_name}")

        mesh_uv = tcmesh.ObjMesh.read_obj(str(obj_fp))

        # We need to collect results across all timepoints
        all_rawvals = []
        all_vals = []
        all_locs = []
        all_args = []

        # Re-using the projection logic
        projected_coordinates = tcinterp.interpolate_per_vertex_field_to_UV(
            mesh_uv,
            mesh_uv.vertices,
            domain="per-vertex",
            uv_grid_steps=uv_grid_steps,
            distance_threshold=0.0000001,
            map_back=True,
            use_fallback="auto",
        )
        projected_normals = tcinterp.interpolate_per_vertex_field_to_UV(
            mesh_uv,
            mesh_uv.normals,
            domain="per-vertex",
            uv_grid_steps=uv_grid_steps,
            distance_threshold=0.0000001,
            map_back=True,
            use_fallback="auto",
        )

        for t_idx, tif_file in enumerate(
            tqdm(tif_files, desc=f"  Projecting {obj_name}")
        ):
            mapping_arr = tifffile.imread(tif_file)
            image = np.expand_dims(mapping_arr, axis=0)

            projected_data = tcinterp.interpolate_volumetric_data_to_uv_multilayer(
                image,
                projected_coordinates,
                projected_normals,
                normal_offsets,
                resolution,
            )

            rawval = np.max(projected_data[0], axis=0)
            argmax = np.argmax(projected_data[0], axis=0)

            loc = projected_coordinates + projected_normals * np.expand_dims(
                normal_offsets[argmax], -1
            )

            val = np.clip(
                (rawval - np.quantile(mapping_arr, 0.5))
                / (np.quantile(mapping_arr, 0.9995) - np.quantile(mapping_arr, 0.5)),
                0,
                1,
            )
            val = np.array(np.rint(val * 255), dtype=np.uint8)
            loc = loc.astype(np.float16)

            all_rawvals.append(rawval)
            all_vals.append(val)
            all_locs.append(loc)
            all_args.append(argmax)

            # Save cellpose stack specifically for this timepoint/mesh
            cp_folder = uv_unwrap_dir / obj_name / "cellpose_stack" / "cellpose"
            cp_folder.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                cp_folder / f"{obj_name}_{t_idx:04d}.tif", np.expand_dims(val, -1)
            )

        # Save aggregated arrays
        print(f"  Saving aggregations for {obj_name}")
        tifffile.imwrite(
            uv_unwrap_dir / f"{obj_name}_all_rawvals.tif",
            np.expand_dims(np.stack(all_rawvals, 0), -1),
        )
        tifffile.imwrite(
            uv_unwrap_dir / f"{obj_name}_all_vals.tif",
            np.expand_dims(np.stack(all_vals, 0), -1),
        )
        tifffile.imwrite(
            uv_unwrap_dir / f"{obj_name}_all_locs.tif", np.stack(all_locs, 0)
        )
        tifffile.imwrite(
            uv_unwrap_dir / f"{obj_name}_all_vals_max_project.tif",
            np.array(np.stack(all_args, 0), dtype=float).astype(np.uint8),
        )

        # Save full space coordinates for reconstruction step
        full_locs = np.expand_dims(projected_coordinates, 0) + np.expand_dims(
            projected_normals, 0
        ) * np.expand_dims(np.array(normal_offsets), [1, 2, 3])
        (uv_unwrap_dir / obj_name / "locs").mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            uv_unwrap_dir / obj_name / "locs" / f"{obj_name}_full_locs.tif", full_locs
        )

        # Save area distortion
        area_distortion = diffgeo.get_area_distortion_in_UV(
            mesh_uv, uv_grid_steps, True
        )
        area_distortion = area_distortion / (uv_grid_steps**2)
        area_distortion = np.array(area_distortion, dtype=np.float32)
        tifffile.imwrite(
            uv_unwrap_dir / f"{obj_name}_area_distortion.tif",
            area_distortion,
            imagej=True,
        )


def run_cellpose_sam(dataset: Path, config: PipelineConfig):
    """
    Runs Cellpose on the 2D projected stacks.
    Adapted from batched/multiprocess_cellpose_dir.py
    """
    print(f"[{dataset.name}] Running CellposeSAM (Cluster)...")

    out_dir = dataset / f"tracking_{config.param_set_name}"
    uv_unwrap_dir = out_dir / "uv_unwrap"

    # We loop over the cellpose_stack subdirectories for each mesh
    for cp_dir in uv_unwrap_dir.glob("*/cellpose_stack/cellpose"):
        obj_name = cp_dir.parent.parent.name
        print(f"  Running Cellpose on {obj_name}")

        tif_files = natsort.natsorted(
            [f for f in cp_dir.iterdir() if f.suffix == ".tif"]
        )

        import torch
        from cellpose import models

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.CellposeModel(
            gpu=config.cluster.cellpose_sam.use_gpu and torch.cuda.is_available(),
            model_type=config.cluster.cellpose_sam.model_type,
            diam_mean=30.0,
            device=device,
        )

        for t_idx, infile in enumerate(tqdm(tif_files, desc="  Segmenting")):
            outfile = cp_dir / f"{infile.stem}_masks.tif"
            if outfile.exists():
                continue

            raw = tifffile.imread(infile)

            # handle axes to ensure strictly 2d
            if raw.ndim == 3 and raw.shape[-1] == 1:
                raw = np.squeeze(raw, axis=-1)

            # typical cellpose expect [cyx] or [yx]
            results = model.eval(
                [raw],
                channels=[0, 0],
                diameter=config.cluster.cellpose_sam.diameter,
                do_3D=False,
                stitch_threshold=0.0,
            )

            masks = np.array(results[0])
            tifffile.imwrite(outfile, masks)

    # Instruct the user to pull back the entire uv_unwrap directory containing the _masks.tif
    print("  Notice: CellposeSAM masks have been saved adjacently to the input tifs.")
