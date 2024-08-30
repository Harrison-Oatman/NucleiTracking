import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import time
import napari
import json
from abc import ABC
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay
from scipy.interpolate import CubicSpline
from tqdm import tqdm
from pathlib import Path


BASE_PATH = Path(r"D:\Tracking\NucleiTracking\data\interim\lightsheet")
SOURCE = 60
LABEL_ITER = 512
PATCH_SIZE = 30
POINT_COUNT = 42


class Patch:
    r = PATCH_SIZE

    def __init__(self, z, y, x, label):
        self.z = round(z)
        self.y = round(y)
        self.x = round(x)
        self.label = label

    def coords(self, img_shape):
        return self._coords(img_shape, [self.r, self.r, self.r])

    def _coords(self, img_shape, lengths):
        z_min = max(0, self.z - lengths[0])
        z_max = min(img_shape[0], self.z + lengths[0])
        y_min = max(0, self.y - lengths[1])
        y_max = min(img_shape[1], self.y + lengths[1])
        x_min = max(0, self.x - lengths[2])
        x_max = min(img_shape[2], self.x + lengths[2])
        return z_min, z_max, y_min, y_max, x_min, x_max

    def normal_coords(self, img_shape, normal):

        axis_lengths = [round(1.5*self.r) for _ in range(3)]

        long_axis = np.argmax(np.abs(normal))
        other_axes = [i for i in range(3) if i != long_axis]
        for i in other_axes:
            axis_lengths[i] = 3*self.r

        return self._coords(img_shape, axis_lengths)

    def check_point(self, z, y, x):
        return (self.z - self.r <= z <= self.z + self.r and
                self.y - self.r <= y <= self.y + self.r and
                self.x - self.r <= x <= self.x + self.r)


class Shape:

    def __init__(self, label, axis, pts=None):
        self.label = label
        self.axis = axis
        if pts is None:
            pts = []
        self.pts = pts

    def project_2d(self):
        if self.axis == 0:
            return np.array(self.pts)[:, 1:]
        elif self.axis == 1:
            return np.array(self.pts)[:, [0, 2]]
        elif self.axis == 2:
            return np.array(self.pts)[:, :2]

    def project_3d(self, pts):
        if self.axis == 0:
            return np.c_[np.ones(len(pts)) * int(self.label), pts[:, 0], pts[:, 1]]
        elif self.axis == 1:
            return np.c_[pts[:, 0], np.ones(len(pts)) * int(self.label), pts[:, 1]]
        elif self.axis == 2:
            return np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts)) * int(self.label)]

    def interpolate(self):
        if len(self.pts) < 2:
            return self.pts
        pts = np.array(self.pts)
        pts = np.concatenate([pts, pts[:1]])

        distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
        print(distances, sum(distances))
        distances = np.concatenate([[0], distances])
        t = np.cumsum(distances) / np.sum(distances)

        cs = CubicSpline(t, pts, bc_type='periodic')
        return cs(np.linspace(0, 1, round(sum(distances) / 40)))

    def make(self):
        if len(self.pts) > 3:
            self.pts = [self.pts[v] for v in ConvexHull(self.project_2d()).vertices]
        return self.pts


def test_point_in_bounds(coords, bounds):
    return all([0 <= c <= b for c, b in zip(coords, bounds)])


def make_shapes_layer(shapes, layer):
    all_shapes = [shape.make() for shape in shapes.values() if len(shape.pts) > 1]
    layer.data = all_shapes
    layer.refresh()


def ray_line_intersection(ray_origin, ray_direction, line):
    a, b = line
    e1 = b - a
    e2 = a - ray_origin
    v = ray_direction
    t2 = (e2[1] * v[0] - e2[0] * v[1]) / (e1[0] * v[1] - e1[1] * v[0])
    if not 0 <= t2 <= 1:
        return False, None

    t1 = (e2[0] + t2 * e1[0]) / v[0]
    if t1 < 0:
        return False, None

    return True, a + t2 * e1


def get_triangle_normal(triangle, theta):
    # theta is used to determine the sign of the normal
    a, b, c = triangle
    e1 = b - a
    e2 = c - a
    n = np.cross(e1, e2)
    if np.dot(n, np.array([np.cos(theta), np.sin(theta), 0])) < 0:
        n = -n
    return n


def y_plane_intercept(y, triangles, theta):
    mins = np.min(triangles[:, :, 1], axis=1)
    maxs = np.max(triangles[:, :, 1], axis=1)
    hits = triangles[np.logical_and(mins <= y, maxs >= y)]
    weights_of_intersection = []
    pts_of_intersection = []
    all_pts = []
    for hit in hits:
        # get the line segments that intersect the y plane
        pts = []
        for i in range(3):
            a, b = hit[i], hit[(i + 1) % 3]
            if (a[1] <= y <= b[1]) or (b[1] <= y <= a[1]):
                if a[1] == b[1]:
                    continue
                mx = (b[0] - a[0]) / (b[1] - a[1])
                x = a[0] + mx * (y - a[1])
                mz = (b[2] - a[2]) / (b[1] - a[1])
                z = a[2] + mz * (y - a[1])
                pts.append([x, z])

        pts = np.array(pts)
        all_pts.append(pts)

        # weigh the intersection by the length of the line segment
        weight = np.linalg.norm(pts[0] - pts[1])
        weights_of_intersection.append(weight)
        pts_of_intersection.append(np.mean(pts, axis=0))

    center = np.sum([w * pt for w, pt in zip(weights_of_intersection, pts_of_intersection)], axis=0) / sum(weights_of_intersection)
    direction = np.array([np.cos(theta), np.sin(theta)])

    for i, pts in enumerate(all_pts):
        intersect, pt = ray_line_intersection(center, direction, pts)
        if intersect:
            normal = get_triangle_normal(hits[i], theta)
            return pt, normal

    return None, None

def ray_triangle_intersection(ray_origin, ray_direction, triangle):
    a, b, c = triangle
    e1 = b - a
    e2 = c - a
    n = np.cross(e1, e2)
    det = -np.dot(ray_direction, n)
    invdet = 1.0 / det
    ao = ray_origin - a
    DAO = np.cross(ao, ray_direction)
    u = np.dot(e2, DAO) * invdet
    v = -np.dot(e1, DAO) * invdet
    t = np.dot(ao, n) * invdet
    if det >= 1e-6 and t >= 0.0 and u >= 0.0 and v >= 0.0 and (u + v) <= 1.0:
        return True, ray_origin + t * ray_direction

    return False, None


def golden_spiral2(n, vertices, triangles):
    y_min = np.min(vertices[:, 1])
    y_max = np.max(vertices[:, 1])

    points = []
    normals = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    triangle_points = np.array([vertices[tri] for tri in triangles])

    for i in tqdm(range(n)):
        y = y_min + (y_max - y_min) * i / n
        new_pt, normal = (y_plane_intercept(y, triangle_points, phi * i))
        if new_pt is None:
            print("warn")
            continue
        points.append([new_pt[0], y, new_pt[1]])
        normals.append(normal)
    return points, normals


# def golden_spiral(n, vertices, triangles):
#     origin_pt = vertices[np.argmin(vertices[:, 1])]
#     final_pt = vertices[np.argmax(vertices[:, 1])]
#     vec = (final_pt - origin_pt)
#     len_max = np.linalg.norm(vec)
#     first_orthog = np.cross(vec, np.array([0, 0, 1]))
#     first_orthog /= np.linalg.norm(first_orthog)
#     second_orthog = np.cross(vec, first_orthog)
#     second_orthog /= np.linalg.norm(second_orthog)
#
#     points = []
#     phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
#
#     triangle_points = np.array([vertices[tri] for tri in triangles])
#
#     for i in tqdm(range(n)):
#         ray_origin = origin_pt + vec * i / n
#
#         theta = phi * i  # golden angle increment
#
#         ray_direction = np.cos(theta) * first_orthog + np.sin(theta) * second_orthog
#
#         for triangle in triangle_points:
#             intersects, point = ray_triangle_intersection(ray_origin, ray_direction, triangle)
#             if intersects:
#                 points.append(point)
#                 break
#
#     return points

def main():
    img = tifffile.imread(BASE_PATH / f"raw\\Recon_fused_tp_{SOURCE}_ch_0_normalized.tif")
    print(img.shape)

    viewer = napari.Viewer()
    img_layer = viewer.add_image(img)
    label_layer = viewer.add_labels(np.zeros_like(img, dtype=np.uint16))
    shapes_layer = viewer.add_shapes()
    shapes_layer_interp = viewer.add_shapes()
    golden_points = viewer.add_points(ndim=3)
    surfaces_layer = viewer.add_surface((np.zeros((3, 3)), np.array([[0, 1, 2]])), colormap='cyan', name='interpolated')

    shapes = {}
    patches = []

    @viewer.bind_key('Control-f')
    def interpolate(layer):
        all_points = []
        for shape in shapes.values():
            if len(shape.pts) < 2:
                continue
            shape_interp = Shape(shape.label, shape.axis, shape.interpolate())
            points = shape_interp.project_3d(shape_interp.project_2d())
            shapes_layer_interp.add(points, shape_type='path')
            all_points.extend(points)

        shapes_layer_interp.refresh()

        c = ConvexHull(all_points)
        surfaces_layer.data = (c.points, c.simplices)
        surfaces_layer.refresh()

        new_points, normals = golden_spiral2(POINT_COUNT, c.points, c.simplices)
        golden_points.add(new_points)

        for point, normal in zip(new_points, normals):
            global LABEL_ITER
            LABEL_ITER += 1
            new_patch = Patch(*point, LABEL_ITER)
            patches.append(new_patch)

            c = new_patch.normal_coords(img.shape, normal)
            label_layer.data[c[0]:c[1], c[2]:c[3], c[4]:c[5]] += new_patch.label

        label_layer.refresh()

        save_path = BASE_PATH / "patch_test" / str(SOURCE) / f"{PATCH_SIZE}_{POINT_COUNT}"
        save_path.mkdir(exist_ok=True, parents=True)
        json_out = {}
        for i, (patch, normal) in enumerate(zip(patches, normals)):
            c = patch.normal_coords(img.shape, normal)
            tifffile.imwrite(str(save_path / f"patch_{i}.tif"), img[c[0]:c[1], c[2]:c[3], c[4]:c[5]])
            json_out[i] = [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]]

        with open(save_path / "patches.json", "w") as f:
            json.dump(json_out, f)

    @shapes_layer.mouse_drag_callbacks.append
    def add_remove_shape(layer, event):

        dragged = False
        yield

        while event.type == 'mouse_move':
            dragged = True
            yield

        if dragged:  # on release
            return

        if viewer.dims.ndisplay == 3:
            return

        coords = layer.world_to_data(event.position)
        critical_axis = viewer.dims.order[0]
        axis_slice = round(viewer.dims.point[critical_axis])
        if len(coords) == 2:
            coords = np.insert(coords, critical_axis, axis_slice)

        if not test_point_in_bounds(coords, img.shape):
            return

        if event.button == 1:  # add patch

            shape = shapes.get(axis_slice, None)
            if shape is None:
                shape = Shape(axis_slice, critical_axis)
                shapes[axis_slice] = shape

            shape.pts.append(coords)
            make_shapes_layer(shapes, shapes_layer)

        elif event.button == 2:  # remove patch
            shape = shapes.get(axis_slice, None)
            if shape is None:
                return
            del shapes[axis_slice]
            make_shapes_layer(shapes, shapes_layer)

    @viewer.bind_key('Control-a')
    def save_shapes(layer):
        print(shapes)
        with open(BASE_PATH / "shapes.json", "w") as f:
            json.dump({k: [list(row) for row in v.pts] for k, v in shapes.items()}, f)

    @viewer.bind_key('Control-d')
    def load_shapes(layer):
        for k in list(shapes.keys()):
            del shapes[k]

        with open(BASE_PATH / "shapes.json", "r") as f:
            shapes.update({int(k): Shape(k, 0, v) for k, v in json.load(f).items()})
        make_shapes_layer(shapes, shapes_layer)
        print(shapes)

    @label_layer.mouse_drag_callbacks.append
    def add_remove_patch(layer, event):

        dragged = False
        yield

        while event.type == 'mouse_move':
            dragged = True
            yield

        if dragged:  # on release
            return


        coords = layer.world_to_data(event.position)

        if event.button == 1:  # add patch
            global LABEL_ITER
            LABEL_ITER += 1

            new_patch = Patch(*coords, LABEL_ITER)
            patches.append(new_patch)

            c = new_patch.coords(img.shape)
            label_layer.data[c[0]:c[1], c[2]:c[3], c[4]:c[5]] += LABEL_ITER
            label_layer.refresh()

        elif event.button == 2:  # remove patch
            for patch in patches:
                if patch.check_point(*coords):  # doesn't work great in 3D
                    patches.remove(patch)
                    c = patch.coords(img.shape)
                    label_layer.data[c[0]:c[1], c[2]:c[3], c[4]:c[5]] -= patch.label
                    label_layer.refresh()

    @viewer.bind_key('k')
    def spread_labels(layer):
        pts = np.zeros((len(patches), 3))

        for i, patch in enumerate(patches):
            pts[i] = [patch.z, patch.y, patch.x]

        pts_new = pts.copy()

        dists = cdist(pts, pts)
        dists[dists == 0] = np.inf
        center = np.mean(pts, axis=0)
        for i, pt in enumerate(pts):
            neighbors = pts[dists[:, i] < 2.5 * Patch.r]
            ds = dists[:, i][dists[:, i] < 2.5 * Patch.r]

            if len(neighbors) == 0:
                continue

            total_vec = np.zeros(3)

            for neighbor, ds in zip(neighbors, ds):
                total_vec += (neighbor - pt) * (ds/Patch.r - 1.5)

            mvt_vector = total_vec / len(neighbors)
            center_vector = center - pt

            mvt_vector = mvt_vector - (mvt_vector @ center_vector) / (center_vector @ center_vector) * center_vector
            pts_new[i] = pt + mvt_vector * 0.45

        label_layer.data = np.zeros_like(label_layer.data, dtype=np.uint16)
        for i, patch in enumerate(patches):
            patch.z, patch.y, patch.x = (round(pt) for pt in pts_new[i])
            c = patch.coords(img.shape)
            label_layer.data[c[0]:c[1], c[2]:c[3], c[4]:c[5]] += patch.label

        label_layer.refresh()

    napari.run()


if __name__ == "__main__":
    main()