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
import io
import shutil


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
    ang_str = f"{int(curr_ang):03d}.{int((curr_ang - int(curr_ang)) * 10):01d}"

    # Output file name
    output_filename = Path(output_dirname) / f"img_ch{ch}_ang{ang_str}_time{t:03d}.tif"

    # Skip existing files if set to do so
    if output_filename.exists():
        logging.info(f"File {output_filename} already exists, skipping.")
        return

    # file e.g.
    with h5py.File(filename, 'r') as h5_file:
        raw_vol = h5_file['/Data'][:]  # stuck here

    logging.info(f"read {filename.name}")
    vol = raw_vol

    # Mirror sheets if from the left camera
    if info['imagingBranch']['image_plane_vectors']['cam_left_to_right'][0] == -1:
        vol = vol[:, ::-1, :]
    if info['imagingBranch']['image_plane_vectors']['cam_left_to_right'][1] == -1:
        vol = vol[:, :, ::-1]

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

    # Save the reformatted image as a .tif file
    tifffile.imwrite(output_filename, vol, dtype=vol.dtype)

    # downscale by 0.5
    vol = skimage.transform.downscale_local_mean(vol, (2, 2, 1))[2:]

    # globally normalize and convert to 16 bit
    data = vol.astype(np.int16)

    # save
    downscaled_outfile = output_dirname / "downscaled" / output_filename.name.replace(".tif", "_downscaled.tif")
    tifffile.imwrite(downscaled_outfile, data)

    mip = np.max(data, axis=0)
    mips_outfile = output_dirname / "mips" / output_filename.name.replace(".tif", "_mip.tif")
    tifffile.imwrite(mips_outfile, mip)

    logging.info(f"saved {output_filename}")


def main():
    args = parse_args()
    logging.basicConfig(level=args.level)

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"directory not found: {input_dir}"

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "downscaled").mkdir(exist_ok=True, parents=True)
    (output_dir / "mips").mkdir(exist_ok=True, parents=True)

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

        for job in jobs:
            job.get()


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
