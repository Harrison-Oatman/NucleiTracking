from tqdm import tqdm
from skimage.measure import regionprops, block_reduce
import pandas as pd
import numpy as np
import skimage.transform as transform

def get_nucleus_df(raw, label_image):
    nucleus_data = isolate_nuclei(raw, label_image)
    nucleus_df = pd.DataFrame(nucleus_data)
    nucleus_df = find_parents(nucleus_df)
    return nucleus_df


def isolate_nuclei(raw, label_image):
    """
    This function takes a label image and returns a dataframe of nuclei parameters
    """

    nucleus_data = {
        "track": [],
        "x": [],
        "y": [],
        "t": [],
        "rad": [],
        "eccentricity": [],
        "orientation": [],
        "label_image": []
    }

    for t, frame in tqdm(enumerate(label_image)):
        for prop in regionprops(frame, raw[t]):
            nucleus_data["track"].append(prop.label)
            nucleus_data["x"].append(prop.centroid[1])
            nucleus_data["y"].append(prop.centroid[0])
            nucleus_data["t"].append(t)
            nucleus_data["rad"].append(prop.axis_major_length / 2)
            nucleus_data["eccentricity"].append(prop.eccentricity)
            nucleus_data["orientation"].append(prop.orientation)
            nucleus_data["label_image"].append(prop.image)

    return nucleus_data


def find_parents(nucleus_df):
    parents = []

    for row in tqdm(nucleus_df.index):
        t, track = nucleus_df["t"][row], nucleus_df["track"][row]
        subset = nucleus_df[nucleus_df["t"] == (t - 1)]
        parent_ind = subset.loc[subset["track"] == track]
        parent = None
        if len(parent_ind.index) > 0:
            parent = parent_ind.index[0]
        parents.append(parent)

    nucleus_df["parent"] = parents

    return nucleus_df


def subimage(img, y, x, rad):
    h = 2 * rad + 1
    ymin = max(0, int(np.floor(y) - rad))
    ymax = min(ymin + h, img.shape[0])
    ymin = min(ymin, ymax - h)
    xmin = max(0, int(np.floor(x) - rad))
    xmax = min(xmin + h, img.shape[1])
    xmin = min(xmin, xmax - h)

    return img[ymin:ymax, xmin:xmax]


def get_normed_intensity_image(nucleus_df, raw, rad=20, ):

    n = nucleus_df.shape[0]
    h = 2 * rad + 1

    imgs = []
    imgs_rot = []

    for i, row in tqdm(enumerate(nucleus_df.index)):
        t, x, y, ori = nucleus_df["t"][row], nucleus_df["x"][row], nucleus_df["y"][row], nucleus_df["orientation"][row]
        img = subimage(raw[t], y, x, rad)
        lo, hi = np.quantile(img, 0.05), np.quantile(img, 0.95)
        img = (img - lo) / (hi - lo)
        img = np.clip(img, 0, 1)
        imgs.append(img)
        imgs_rot.append(transform.rotate(img, np.rad2deg(-ori)))

    nucleus_df["img"] = [arr for arr in imgs]
    nucleus_df["img_rot"] = [arr for arr in imgs_rot]
    nucleus_df["img_reduced"] = [block_reduce(img, 2, np.mean) for img in nucleus_df["img"]]

    return nucleus_df
