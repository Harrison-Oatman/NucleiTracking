import skimage
import argparse
import logging
import multiprocessing
from pathlib import Path
import os
import re
import json
import numpy as np
import h5py
import tifffile


def reconstruct(filename, output_dirname, sd, ch, t):

    logging.info(f"processing {filename}")
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(f"file not found: {filename}")

    # Load metadata from corresponding .json file
    info_filename = str(filename).replace('.lux.h5', '.json')
    with open(info_filename, 'r') as f:
        info = json.load(f)

    # Determine acquisition angle
    elements = info['metaData']['stack']['elements']
    if isinstance(elements, list):
        curr_ang = elements[3]['start']
    else:
        curr_ang = elements[4].get('start')

    if sd.lower() == "right":
        curr_ang = (curr_ang + 180) % 360
    ang_str = f"{int(curr_ang):03d}.0"

    # Output file name
    output_filename = Path(output_dirname) / f"img_ch{ch}_ang{ang_str}_time{t:03d}.tif"
    downscaled_outfile = output_dirname / "downscaled" / output_filename.name.replace(".tif", "_downscaled.tif")

    # Skip existing files if set to do so
    if downscaled_outfile.exists():
        logging.info(f"File {downscaled_outfile} already exists, skipping.")
        return

    try:
        # file e.g.
        with h5py.File(filename, 'r') as h5_file:
            raw_vol = h5_file['/Data'][:]

    except OSError:
        logging.error(f"EOFError: {filename} is empty or corrupted.")
        return downscaled_outfile.stem

    logging.info(f"read {filename.name}")
    vol = np.array(raw_vol).astype(np.int16)

    # Mirror sheets if from the left camera
    if info['imagingBranch']['image_plane_vectors']['cam_left_to_right'][0] == -1:
        vol = vol[:, :, ::-1]
    if info['imagingBranch']['image_plane_vectors']['cam_left_to_right'][1] == -1:
        vol = vol[:, ::-1, :]

    # Reverse Z values if sheets acquired from higher to lower Z
    z_elements = elements[2] if isinstance(elements, list) else elements.get(3)
    is_z_reversed = (z_elements['end'] - z_elements['start']) < 0
    if is_z_reversed:
        vol = vol[::-1, :, :]

    spacing = np.array(list(info['processingInformation']['voxel_size_um'].values()))
    spacing_str = ' '.join(map(lambda x: f"{x:.6f}", spacing))
    spacing_filename = os.path.join(output_dirname, f"spacing {spacing_str}.txt")
    with open(spacing_filename, 'w') as f:
        f.write('')

    # save the json file
    json_filename = output_dirname / "json" / output_filename.name.replace(".tif", ".json")
    with open(json_filename, 'w') as f:
        json.dump(info, f, indent=4)

    # Save the reformatted image as a .tif file
    # tifffile.imwrite(output_filename, vol, dtype=vol.dtype)

    out_dtype = vol.dtype

    # downscale by 0.5
    vol = skimage.transform.downscale_local_mean(vol, (1, 2, 2))

    # convert to int 16
    vol = np.floor(vol).astype(np.uint16)

    # save
    downscaled_outfile = output_dirname / "downscaled" / output_filename.name.replace(".tif", "_downscaled.tif")
    tifffile.imwrite(downscaled_outfile, vol, dtype=np.uint16)

    mip = np.max(vol, axis=0)
    mips_outfile = output_dirname / "mips" / output_filename.name.replace(".tif", "_mip.tif")
    tifffile.imwrite(mips_outfile, mip, dtype=np.uint16)

    logging.info(f"saved {output_filename}")

    return None


def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "downscaled").mkdir(exist_ok=True, parents=True)
    (output_dir / "mips").mkdir(exist_ok=True, parents=True)
    (output_dir / "json").mkdir(exist_ok=True, parents=True)

    nprocs = args.nprocs

    # Parsing filenames to extract file identities
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    pattern = re.compile(r'^stack_(\d+)_channel_(\d+)_obj_(left|right)$')

    tok = [pattern.match(d).groups() for d in subdirs if pattern.match(d)]

    stacks = sorted(set(int(t[0]) for t in tok))
    channels = sorted(set(int(t[1]) for t in tok))
    side = sorted(set(t[2].capitalize() for t in tok))

    logging.info(f"Found stacks {stacks}, channels {channels}, and sides {side}.")

    with multiprocessing.Pool(processes=nprocs) as pool:
        jobs = []
        for s in stacks:
            for ch in channels:
                for sd in side:
                    stack_id = f"stack_{s}_channel_{ch}_obj_{sd.lower()}"
                    curr_dir = Path(input_dir) / stack_id

                    # Get a list of time points in the folder
                    curr_files = list(curr_dir.glob("*.h5"))

                    if not curr_files:
                        continue

                    time_pattern = re.compile(f"Cam_{sd.lower()}_(\\d+).lux.h5")
                    time_points = [int(time_pattern.match(f.name).group(1)) for f in curr_files if time_pattern.match(f.name)]

                    logging.info(f"Detected {len(time_points)} time points for {stack_id}.")

                    filepaths = [curr_dir / f"Cam_{sd.lower()}_{t:05d}.lux.h5" for t in sorted(time_points)]

                    for f, t in zip(filepaths, sorted(time_points)):
                        jobs.append(pool.apply_async(reconstruct, (f, output_dir, sd, ch, t)))

        failed = []

        for job in jobs:
            out = job.get()

            if out is not None:
                failed.append(out)

    print("DONE")

    if failed:
        logging.error(f"Failed to process the following files: {failed}")


def parse_args():
    parser = argparse.ArgumentParser(description="script to downscale and maxproject tifs from a trajectory")
    parser.add_argument("-i", "--input_dir", help="process all tifs in directory", default=None)
    parser.add_argument("-o", "--output", help="results directory", default=None)
    parser.add_argument("-l", "--level", default="INFO")
    parser.add_argument("--nprocs", help="number of processes", default=None, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
