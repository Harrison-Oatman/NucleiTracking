import h5py
from pathlib import Path
import multiprocessing
import logging


def main(f):
    filename = Path(f)

    with h5py.File(filename, 'r') as h5_file:
        raw_vol = h5_file['/Data'][:]

    logging.info(raw_vol.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    jobs = []

    with multiprocessing.Pool(processes=4) as pool:
        filepaths = list(Path(f"/mnt/ceph/users/hoatman/lightsheet_20241030/raw/stack_0_channel_0_obj_left/").glob("*.h5"))
        for f in filepaths:
            jobs.append(pool.apply_async(main, (str(f), )))

        for job in jobs:
            job.get()