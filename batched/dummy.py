import h5py
from pathlib import Path
import multiprocessing
import logging


def main(i):
    filename = Path(f"/mnt/ceph/users/hoatman/lightsheet_20241030/raw/stack_0_channel_0_obj_left/Cam_left_0000{i}.lux.h5")

    with h5py.File(filename, 'r') as h5_file:
        raw_vol = h5_file['/Data'][:]

    logging.info(raw_vol.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    jobs = []

    with multiprocessing.Pool(processes=4) as pool:
        for i in range(10):
            jobs.append(pool.apply_async(main, (i,)))

        for job in jobs:
            job.get()