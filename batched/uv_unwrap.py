import argparse
import logging
import tifffile
import numpy as np
from tqdm import tqdm
import time
import multiprocessing
from pathlib import Path
import natsort
from blender_tissue_cartography import mesh as tcmesh
from blender_tissue_cartography import interpolation as tcinterp


def main():
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    print("starting main")
    args = process_cli()

    logging.basicConfig(level=args.level)

    inpath = args.input_dir
    inpath = Path(inpath)
    if not inpath.exists():  # try relative path
        inpath = Path().cwd() / args.input

    assert inpath.exists(), f"directory not found: {inpath}"
    print(f"processing files in {inpath}")

    outpath = args.output
    outpath = Path(outpath) if outpath is not None else inpath.parent / f"uv_unwrap"
    outpath.mkdir(exist_ok=True)

    for obj in Path(args.obj).glob("*.obj"):
        name = obj.stem
        (outpath / name / "vals").mkdir(exist_ok=True, parents=True)
        (outpath / name / "locs").mkdir(exist_ok=True, parents=True)

    files = natsort.natsorted([f for f in inpath.iterdir() if f.suffix == '.tif'])
    print(f"found {len(files)} tif files")

    nprocs = args.nprocs

    start = time.time()

    with multiprocessing.Pool(processes=nprocs) as pool:
        print(f"pool initialized in {time.time() - start} seconds")

        jobs = []
        for i, file in tqdm(enumerate(files)):
            job = pool.apply_async(process_file, (i, str(file.absolute()), args, str(outpath.absolute())))
            jobs.append(job)

        vals_and_locs = [job.get() for job in jobs]

    for obj in Path(args.obj).glob("*.obj"):
        name = obj.stem

        vals = [v[name] for v, _, _ in vals_and_locs]
        locs = [l[name] for _, l, _ in vals_and_locs]
        maxp = [m[name] for _, _, m in vals_and_locs]


        v_stack = np.stack(vals, 0)
        l_stack = np.stack(locs, 0)
        maxp_stack = np.array(np.array(np.stack(maxp, 0), dtype=float), dtype=np.uint8)

        tifffile.imwrite(outpath / f"{name}_all_vals.tif", v_stack)
        tifffile.imwrite(outpath / f"{name}_all_locs.tif", l_stack)
        tifffile.imwrite(outpath / f"{name}_all_vals_max_project.tif", maxp_stack)


def process_cli() -> argparse.Namespace:
    argparser = argparse.ArgumentParser(description="script to process raw data from tif")

    argparser.add_argument("-i", "--input_dir", dest="input_dir", help="path to raw file to process", default=None)
    argparser.add_argument("-o", "--output", dest="output", help="results directory", default=None)
    argparser.add_argument("--obj", help="path to blender object file directory", default=None)

    argparser.add_argument_group("box")
    argparser.add_argument("--range", nargs=3, type=int, default=[0, 0, 1])
    argparser.add_argument("--resolution", nargs=3, type=int, default=[1, 1, 1])
    argparser.add_argument("--uv_grid_steps", type=int, default=1024)

    argparser.add_argument_group("multiprocessing")
    argparser.add_argument("--nprocs", default=1, type=int)

    argparser.add_argument_group("other")
    argparser.add_argument("-l", "--level", default="INFO")

    return argparser.parse_args()


def process_file(j, infile, args, outpath):
    logging.basicConfig(level=args.level)

    infile = Path(infile)
    outpath = Path(outpath)

    logging.info(f"processing file {infile.stem} on iter {j}")

    mapping_arr = tifffile.imread(str(infile))

    out = {}

    for obj_fp in Path(args.obj).glob("*.obj"):

        obj_name = obj_fp.stem

        mesh_uv = tcmesh.ObjMesh.read_obj(args.obj)
        normal_offsets = np.linspace(args.range[0], args.range[1], args.range[2])

        mesh = mesh_uv
        image = np.expand_dims(mapping_arr, axis=0)
        uv_grid_steps = args.uv_grid_steps
        map_back = True
        use_fallback = "auto"
        resolution = (args.resolution[0], args.resolution[1], args.resolution[2])

        projected_coordinates = tcinterp.interpolate_per_vertex_field_to_UV(mesh, mesh.vertices, domain="per-vertex",
                                                                            uv_grid_steps=uv_grid_steps,
                                                                            distance_threshold=0.0000001,
                                                                            map_back=map_back,
                                                                            use_fallback=use_fallback)
        projected_normals = tcinterp.interpolate_per_vertex_field_to_UV(mesh, mesh.normals, domain="per-vertex",
                                                                        uv_grid_steps=uv_grid_steps,
                                                                        distance_threshold=0.0000001,
                                                                        map_back=map_back, use_fallback=use_fallback)
        projected_data = tcinterp.interpolate_volumetric_data_to_uv_multilayer(image,
                                                                               projected_coordinates,
                                                                               projected_normals, normal_offsets,
                                                                               resolution)

        val = np.max(projected_data[0], axis=0)
        argmax = np.argmax(projected_data[0], axis=0)

        loc = projected_coordinates + projected_normals * np.expand_dims(normal_offsets[argmax], -1)

        # convert to 32-bit float
        val = val.astype(np.float32)
        loc = loc.astype(np.float16)

        val_outfile = outpath / obj_name / "vals" / f"{infile.stem}_unwrap.tif"
        loc_outfile = outpath / obj_name / "locs" / f"{infile.stem}_unwrap_locs.tif"

        # save the results
        tifffile.imwrite(val_outfile, val)
        tifffile.imwrite(loc_outfile, np.array(loc))

        out[obj_name] = (val, loc, argmax)

    return out


if __name__ == "__main__":
    main()
