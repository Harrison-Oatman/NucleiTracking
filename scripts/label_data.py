import cv2
import time
import msvcrt as m
from pathlib import Path
import tifffile
import numpy as np
from skimage.measure import regionprops, regionprops_table
import pandas as pd

MASK_PATH = Path(r"D:\Tracking\NucleiTracking\data\interim\cellpose_out\embryo014a_MaxIP_bgs_crop007masks.tif")
RAW_PATH = Path(r"D:\Tracking\NucleiTracking\data\interim\confocal\embryo014a\embryo014a_MaxIP_bgs.tif")

# EXPORT_PATH = Path(r"D:\Tracking\NucleiTracking\data\processed") / time.strftime("%Y%m%d%H%M%S")
EXPORT_PATH = Path(r"D:\Tracking\NucleiTracking\data\processed\embryo014a")

SLICES = [5, 10, 11, 12, 13, 21, 22, 41, 51, 54, 64, 67, 78, 100, 107, 114, 119, 168, 187, 189, 198, 210]
SPOTS_PER_SLICE = 25

CROP_SIZE = 48
BIG_BOX_SIZE = 256
TRAIN_TEST_SPLIT = 0.8

LABELS = {
    '1': 'interphase',
    '2': 'prophase',
    '3': 'metaphase',
    '4': 'anaphase',
    '5': 'other',
    '-': 'unlabelled'
}

CONTROLS = {
    'q': 'quit',
    'd': 'next',
    'a': 'previous',
    's': 'save',
    'w': 'next_unlabelled'
}


def crop(img, y, x, size):
    return img[y - size // 2:y + size // 2, x - size // 2:x + size // 2]


def label_display_loop(spot):
    img = spot["big_crop"]

    # scale up image
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    # rescale brightness
    imin, imax = np.quantile(img, 0.01), np.quantile(img, 0.99)
    img = np.clip((img - imin) / (imax - imin), 0, 1) * 255
    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # draw box around small crop in center
    cv2.rectangle(img, (img.shape[1] // 2 - CROP_SIZE // 2, img.shape[0] // 2 - CROP_SIZE // 2),
                  (img.shape[1] // 2 + CROP_SIZE // 2, img.shape[0] // 2 + CROP_SIZE // 2),
                  color=(0, 0, 255),thickness=1)

    cv2.imshow(f"spot: {LABELS[spot['label']]}", img)

    key = cv2.waitKey(0)

    if key in [ord(k) for k in CONTROLS.keys()]:
        cv2.destroyAllWindows()
        return spot["label"], CONTROLS[chr(key)]

    if key in [ord(k) for k in LABELS.keys()]:
        cv2.destroyAllWindows()
        return chr(key), 'next'

    else:
        cv2.destroyAllWindows()
        return label_display_loop(spot)


def import_csv(masks, raw, path):
    data = pd.read_csv(path)

    crops = []
    big_crops = []

    for i, spot in data.iterrows():
        img = raw[spot["slice"]]
        crops.append(crop(img, round(spot["centroid-0"]), round(spot["centroid-1"]), CROP_SIZE))
        big_crops.append(crop(img, round(spot["centroid-0"]), round(spot["centroid-1"]), BIG_BOX_SIZE))

    data["crop"] = crops
    data["big_crop"] = big_crops

    return data


def new_csv(masks, raw):
    data = []

    for z in SLICES:
        mask = masks[z]
        img = raw[z]

        props = pd.DataFrame(regionprops_table(mask, properties=('label', 'centroid')))
        props["slice"] = [z] * len(props["label"])
        props["train"] = np.random.rand(len(props["label"])) < TRAIN_TEST_SPLIT
        props["label"] = ['-'] * len(props["label"])

        crops = []
        big_crops = []
        include = []
        for i, row in props.iterrows():
            crops.append(crop(img, round(row["centroid-0"]), round(row["centroid-1"]), CROP_SIZE))
            big_crops.append(crop(img, round(row["centroid-0"]), round(row["centroid-1"]), BIG_BOX_SIZE))

            include.append(big_crops[-1].shape == (BIG_BOX_SIZE, BIG_BOX_SIZE))

        props["crop"] = crops
        props["big_crop"] = big_crops
        props = props[include].sample(SPOTS_PER_SLICE)

        data.append(props)

    return pd.concat(data, ignore_index=True)


def save(data):
    print("saving")
    labeled = data[data["label"] != '-']
    save_csv = data[[key for key in data.keys() if key not in ["crop", "big_crop"]]]
    save_csv["filepath"] = [None] * len(save_csv)

    for i, spot in labeled.iterrows():
        train_label = "train" if spot["train"] else "test"
        img_path = str(EXPORT_PATH / f"{i}_{train_label}_{LABELS[spot['label']]}.png")
        cv2.imwrite(img_path, spot["crop"])
        save_csv.loc[i, "filepath"] = img_path

    save_csv.to_csv(EXPORT_PATH / "labels.csv", index=True)

def main():
    masks = tifffile.imread(MASK_PATH)
    raw = tifffile.imread(RAW_PATH)

    if not EXPORT_PATH.exists():
        EXPORT_PATH.mkdir()

    if (EXPORT_PATH / "labels.csv").exists():
        data = import_csv(masks, raw, EXPORT_PATH / "labels.csv")
        print("loaded existing labels.csv")

    else:
        data = new_csv(masks, raw)
        print("created new labels.csv")

    n = len(data["label"])

    i = 0
    while True:
        spot = data.iloc[i]
        label, action = label_display_loop(spot)
        data.at[i, "label"] = label

        if action == 'quit':
            break
        elif action == 'next':
            i += 1
        elif action == 'previous':
            i -= 1
        elif action == 'save':
            save(data)
        elif action == 'next_unlabelled':
            i = (i + 1) % n
            while data.at[i, "label"] != '-':
                i = (i + 1) % n

        i = i % n


if __name__ == "__main__":
    main()