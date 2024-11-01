import h5py
from pathlib import Path


def main():
    filename = Path(r"/mnt/ceph/users/hoatman/lightsheet_20241030/raw/stack_0_channel_0_obj_left/Cam_left_00000.lux.h5")

    with h5py.File(filename, 'r') as h5_file:
        raw_vol = h5_file['/Data'][:]

    print(raw_vol.shape)


if __name__ == "__main__":
    main()