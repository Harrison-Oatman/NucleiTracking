from pathlib import Path

import numpy as np
import pymeshlab
import tifffile
from blender_tissue_cartography import interface_pymeshlab as intmsl
from blender_tissue_cartography import mesh as tcmesh
from scipy.stats import mode
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians
from sklearn.cluster import DBSCAN

from nucleitracking.pipeline.config import PipelineConfig


def run_peak_detection(dataset: Path, config: PipelineConfig):
    """
    Runs Difference of Gaussians to detect peak local maxima
    (nuclei centers) on the final timepoint.
    """
    out_dir = dataset / f"tracking_{config.param_set_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dog_peaks.npy"
    if out_file.exists():
        print(f"[{dataset.name}] Peak Detection (Skipped: {out_file.name} exists)")
        return

    print(f"[{dataset.name}] Running Peak Detection...")

    # Locate the final timepoint image
    # Assuming 'recon' folder contains the deconvolved images
    recon_dir = dataset / "recon"
    if not recon_dir.exists():
        print(f"  [Error] Could not find 'recon' directory at {recon_dir}")
        return

    images = list(recon_dir.glob("*.tif"))
    if not images:
        print(f"  [Error] No .tif images found in {recon_dir}")
        return

    # Sort to get the final timepoint
    images.sort()
    final_img_path = images[-1]
    print(f"  Using image: {final_img_path.name}")

    arr = tifffile.imread(final_img_path)

    # Parameters could be pulled from config in the future
    sigma_low = config.local_pre.peak_detection.model_extra.get("sigma_low", 2)
    sigma_high = config.local_pre.peak_detection.model_extra.get("sigma_high", 6)
    min_distance = config.local_pre.peak_detection.model_extra.get("min_distance", 5)
    threshold_abs = config.local_pre.peak_detection.model_extra.get("threshold_abs", 35)

    print("  Applying Difference of Gaussians...")
    dog = difference_of_gaussians(arr, sigma_low, sigma_high)

    print("  Finding peaks...")
    peaks = peak_local_max(dog, min_distance=min_distance, threshold_abs=threshold_abs)

    np.save(out_file, peaks)
    print(f"  Saved {len(peaks)} peaks to {out_file}")


def run_mesh_generation(dataset: Path, config: PipelineConfig):
    """
    Uses DBSCAN to cluster peaks, keeping the largest cluster (the embryo),
    and reconstructs a surface mesh using Poisson reconstruction.
    """
    out_dir = dataset / f"tracking_{config.param_set_name}"
    obj_out_path = out_dir / "dog_peaks.obj"
    if obj_out_path.exists():
        print(f"[{dataset.name}] Mesh Generation (Skipped: {obj_out_path.name} exists)")
        return

    print(f"[{dataset.name}] Running Mesh Generation...")

    peaks_path = out_dir / "dog_peaks.npy"
    if not peaks_path.exists():
        print(
            f"  [Error] Peaks file not found at {peaks_path}. Run peak detection first."
        )
        return

    peaks = np.load(peaks_path)

    print("  Clustering peaks with DBSCAN...")
    dbscan = DBSCAN(eps=15, min_samples=1)
    vals = dbscan.fit_predict(peaks)

    # Keep the largest cluster
    vmax = mode(vals, keepdims=False).mode
    filtered_peaks = peaks[vals == vmax]
    print(f"  Retained {len(filtered_peaks)} peaks in the main cluster.")

    print("  Reconstructing surface mesh using PyMeshLab...")
    point_cloud = tcmesh.ObjMesh(vertices=filtered_peaks, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)

    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_screened_poisson(
        depth=8,
        fulldepth=5,
    )

    ms.meshing_isotropic_explicit_remeshing(
        iterations=10, targetlen=pymeshlab.PercentageValue(1)
    )

    mesh_reconstructed = intmsl.convert_from_pymeshlab(ms.current_mesh())

    obj_out_path = out_dir / "dog_peaks.obj"
    mesh_reconstructed.write_obj(str(obj_out_path))
    print(f"  Saved reconstructed mesh to {obj_out_path}")


def run_uv_unwrapping(dataset: Path, config: PipelineConfig):
    """
    UV Unwrapping is a manual step performed in Blender.
    This function acts as a checkpoint to verify the user has
    completed the manual step and placed the generated pieces in the correct folder.
    """
    print(f"[{dataset.name}] Checking UV Unwrapping...")

    out_dir = dataset / f"tracking_{config.param_set_name}"
    uv_unwrap_dir = out_dir / "uv_unwrap"

    if not uv_unwrap_dir.exists() or not list(uv_unwrap_dir.glob("*.obj")):
        print(f"  [Warning] No '.obj' UV files found in {uv_unwrap_dir}.")
        print("  ACTION REQUIRED:")
        print("  1. Import 'dog_peaks.obj' into Blender.")
        print(
            "  2. Manually cut seams and unwrap the mesh (e.g. into top, bottom, poles)."
        )
        print("  3. Export the unwrapped meshes as .obj files (with UVs included).")
        print(f"  4. Place the exported .obj files in: {uv_unwrap_dir}")
    else:
        obj_files = list(uv_unwrap_dir.glob("*.obj"))
        print(f"  Found {len(obj_files)} UV unwrapped meshes. Ready for cluster phase.")
