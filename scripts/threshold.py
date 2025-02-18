import napari
import numpy as np
from skimage.filters import difference_of_gaussians, rank
from skimage.feature import peak_local_max
from scipy.signal import convolve
# from skimage.morphology import sp
import pandas as pd
import os
import tifffile as tiff
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel, QPushButton, QSlider, QComboBox
from natsort import natsorted
import json
from typing import NamedTuple
from pathlib import Path


class JsonStoreValue:

    def __init__(self, jsonpath):

        self.jsonpath = jsonpath

        with open(jsonpath, 'r') as f:
            self.json = json.load(f)

    def save(self):
        with open(self.jsonpath, "w") as f:
            json.dump(self.json, f)
        print("saved")

    def update(self, stem, preset: dict):
        preset["dog"] = str(preset["dog"])
        preset["mind"] = str(preset["mind"])
        preset["value"] = int(preset["value"])
        preset["local_value"] = int(preset["local_value"])
        self.json[stem] = preset

        self.save()

    def get(self, stem):
        return self.json.get(stem)


class ThresholdSlider(QWidget):
    def __init__(self, update_callback, store_value_callback, dogs, mins, min_val=25, max_val=250):
        super().__init__()
        self.update_callback = update_callback
        self.store_value_callback = store_value_callback
        self.layout = QVBoxLayout()
        self.threshold_value = 0
        self.local_threshold_value = 0

        self.dog_label = QLabel("DOG preset")
        self.dog_combobox = QComboBox()
        self.dog_combobox.addItems(dogs)
        self.dog_combobox.currentTextChanged.connect(self.on_update)

        self.dis_label = QLabel("Min Distance")
        self.dis_combobox = QComboBox()
        self.dis_combobox.addItems(mins)
        self.dis_combobox.currentTextChanged.connect(self.on_update)

        self.threshold_label = QLabel(f"Threshold: {min_val}")
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Vertical slider
        self.threshold_slider.setMinimum(min_val)
        self.threshold_slider.setMaximum(max_val)
        self.threshold_slider.setValue(min_val)
        self.threshold_slider.valueChanged.connect(self.update_slider)

        self.local_threshold_label = QLabel(f"Local Threshold: {min_val}")
        self.local_threshold_slider = QSlider()
        self.local_threshold_slider.setOrientation(1)  # Vertical slider
        self.local_threshold_slider.setMinimum(min_val)
        self.local_threshold_slider.setMaximum(max_val)
        self.local_threshold_slider.setValue(min_val)
        self.local_threshold_slider.valueChanged.connect(self.update_local_slider)

        self.store_value_button = QPushButton("Store Value")
        self.store_value_button.clicked.connect(self.store_value)

        self.layout.addWidget(self.dog_label)
        self.layout.addWidget(self.dog_combobox)
        self.layout.addWidget(self.dis_label)
        self.layout.addWidget(self.dis_combobox)
        self.layout.addWidget(self.threshold_label)
        self.layout.addWidget(self.threshold_slider)
        self.layout.addWidget(self.local_threshold_label)
        self.layout.addWidget(self.local_threshold_slider)
        self.setLayout(self.layout)

    def update_slider(self, value):
        self.threshold_value = value
        self.threshold_label.setText(f"Threshold: {value}")

        self.on_update(None)

    def update_local_slider(self, value):
        self.local_threshold_value = value
        self.local_threshold_label.setText(f"Local Threshold: {value}")

        self.on_update(None)

    def set_status(self, value, local_value, dog, mind):
        self.dog_combobox.setCurrentText(dog)
        self.dis_combobox.setCurrentText(mind)
        self.threshold_slider.setValue(value)
        self.local_threshold_slider.setValue(local_value)

        self.on_update(None)

    def get_status(self):
        value = int(self.threshold_value)
        dog_preset = self.dog_combobox.currentText()
        dis_preset = self.dis_combobox.currentText()
        local_value = int(self.local_threshold_value)
        return value, local_value, dog_preset, dis_preset

    def store_value(self):
        self.store_value_callback()

    def on_update(self, _):
        v, local_v, dog, mind = self.get_status()
        self.update_callback(v, local_v, dog, mind)


# def get_peaks(img, sigma_hi, sigma_low, low_threshold=20, min_distance=3):
#     v = difference_of_gaussians(img, sigma_low, sigma_hi)
#     # v2 = convolve(v, np.ones((5, 5, 5)) / 5**3, mode="same")
#     pts = peak_local_max(v, min_distance=min_distance, threshold_abs=low_threshold)
#     # vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]
#     vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]
#     print(np.mean(vals))
#
#     return pd.DataFrame({"z": pts[:, 0], "y": pts[:, 1], "x": pts[:, 2], "val": vals})


def load_tiffs_and_peaks(directory, peak_directory):
    tiff_files = natsorted([f for f in os.listdir(directory) if f.endswith('.tiff') or f.endswith('.tif')])
    csv_files = natsorted([f for f in os.listdir(peak_directory) if f.endswith('.csv')])

    images = [tiff.imread(os.path.join(directory, f)) for f in tiff_files]
    images = [np.swapaxes(image, 0, 2) for image in images]
    peak_data = [pd.read_csv(os.path.join(peak_directory, f)) for f in csv_files]

    stems = [Path(f).stem for f in csv_files]

    return images, peak_data, stems


class GuiProcess:

    def __init__(self, directory, peak_directory, jsonpath):

        self.directory = directory
        self.current_frame = 0
        self.image_data, self.peak_data_list, self.stems = load_tiffs_and_peaks(directory, peak_directory)

        jsonstorevalue = JsonStoreValue(jsonpath)
        self.store_value_callback = jsonstorevalue.update
        self.get_preset_callback = jsonstorevalue.get

        self.dogs = self.peak_data_list[0]["dog"].unique()
        self.minds = [str(d) for d in self.peak_data_list[0]["min-distance"].unique()]

        self.viewer = napari.Viewer(ndisplay=3)
        self.image_layer = None
        self.scatter_layer = None

        self.slider = ThresholdSlider(self.update_peaks, self.store_preset, self.dogs, self.minds)
        self.viewer.window.add_dock_widget(self.slider, area='right')

        # Bind keys for manual frame navigation
        self.viewer.bind_key("Right", lambda viewer: self.change_frame(1))
        self.viewer.bind_key("Left", lambda viewer: self.change_frame(-1))
        self.viewer.bind_key("shift+s", lambda viewer: self.store_preset())

    def change_frame(self, step):
        new_frame = self.current_frame + step

        # Ensure frame index stays within bounds
        if 0 <= new_frame < len(self.image_data):
            self.current_frame = new_frame
            self.update_frame()

    def store_preset(self):
        value, local_value, dog, mind = self.slider.get_status()
        stem = self.stems[self.current_frame]
        self.store_value_callback(stem, {"value": value, "local_value": local_value, "dog": dog, "mind": mind})

    def update_frame(self):
        """Manually update frame display"""
        self.image_layer.data = self.image_data[self.current_frame]

        if self.get_preset_callback(self.stems[self.current_frame]) is not None:
            preset = self.get_preset_callback(self.stems[self.current_frame])
            dog = preset["dog"]
            mind = preset["mind"]
            value = preset["value"]
            local_value = preset["local_value"]
            self.slider.set_status(value, local_value, dog, mind)

        else:
            value, local_value, dog, mind = self.slider.get_status()
            self.update_peaks(value, local_value, dog, mind)

        # self.scatter_layer.data = self.peak_data_list[self.current_frame][['x', 'y', 'z']].values

    def run(self):

        self.image_layer = self.viewer.add_image(self.image_data[0], name='3D Image')
        self.scatter_layer = self.viewer.add_points(self.peak_data_list[0][['x', 'y', 'z']].values, shading="spherical",
                                                    name='Peaks', size=10, face_color='red', out_of_slice_display=True)

        self.update_frame()

        napari.run()

    def update_peaks(self, value, local_value, dog_preset, min_distance_preset):

        data = self.peak_data_list[self.current_frame]
        print(len(data))
        data = data[data["dog"] == dog_preset]
        print(len(data))
        data = data[data["min-distance"].astype(str) == min_distance_preset]
        print(len(data))
        data = data[data['val'] > value]
        print(len(data))
        data = data[data['local'] > local_value]
        print(len(data))

        filtered_peaks = data[['x', 'y', 'z']].values
        self.scatter_layer.data = filtered_peaks

    # def recompute_peaks(self, sigma_high, sigma_low, min_distance):
    #     img = self.image_data[self.current_frame]
    #     peaks = get_peaks(img, sigma_high, sigma_low, min_distance=min_distance)
    #     self.peak_data_list[self.current_frame] = peaks
    #     self.update_frame()


def main():

    directory = r"D:\Tracking\NucleiTracking\data\interim\lightsheet\2025_02_06\recon\test2"
    jsonpath = r"D:\Tracking\NucleiTracking\data\interim\lightsheet\2025_02_06\recon\test2\presets.json"

    directory = r"/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/recon/"
    peak_directory = r"/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/recon/dog_sweep"
    jsonpath = r"/mnt/home/hoatman/ceph/lightsheet_20250206/raw_image/downscaled/recon/presets.json"

    if not Path(jsonpath).exists():
        with open(jsonpath, 'w') as f:
            json.dump({}, f)

    gui_process = GuiProcess(directory, peak_directory, jsonpath)
    gui_process.run()


if __name__ == "__main__":
    main()
