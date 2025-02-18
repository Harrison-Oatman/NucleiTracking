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


class ThresholdSlider(QWidget):
    def __init__(self, update_callback, recompute_callback, min_val=0, max_val=150):
        super().__init__()
        self.update_callback = update_callback
        self.recompute_callback = recompute_callback
        self.layout = QVBoxLayout()

        self.sigma_high_label = QLabel("Sigma High:")
        self.sigma_high_input = QLineEdit()
        self.sigma_high_input.setText("5")

        self.sigma_low_label = QLabel("Sigma Low:")
        self.sigma_low_input = QLineEdit()
        self.sigma_low_input.setText("2")

        self.min_distance_label = QLabel("Min Distance:")
        self.min_distance_input = QLineEdit()
        self.min_distance_input.setText("3")

        self.threshold_label = QLabel(f"Threshold: {min_val}")
        self.threshold_slider = QSlider()
        self.threshold_slider.setOrientation(1)  # Vertical slider
        self.threshold_slider.setMinimum(min_val)
        self.threshold_slider.setMaximum(max_val)
        self.threshold_slider.setValue(min_val)
        self.threshold_slider.valueChanged.connect(self.on_threshold_change)

        self.recompute_button = QPushButton("Recompute")
        self.recompute_button.clicked.connect(self.on_recompute)

        self.layout.addWidget(self.sigma_high_label)
        self.layout.addWidget(self.sigma_high_input)
        self.layout.addWidget(self.sigma_low_label)
        self.layout.addWidget(self.sigma_low_input)
        self.layout.addWidget(self.min_distance_label)
        self.layout.addWidget(self.min_distance_input)
        self.layout.addWidget(self.threshold_label)
        self.layout.addWidget(self.threshold_slider)
        self.layout.addWidget(self.recompute_button)
        self.setLayout(self.layout)

    def on_threshold_change(self, value):
        self.threshold_label.setText(f"Threshold: {value}")
        self.update_callback(value)

    def on_recompute(self):
        sigma_high = float(self.sigma_high_input.text())
        sigma_low = float(self.sigma_low_input.text())
        min_distance = int(self.min_distance_input.text())
        self.recompute_callback(sigma_high, sigma_low, min_distance)


def get_peaks(img, sigma_hi, sigma_low, low_threshold=20, min_distance=3):
    v = difference_of_gaussians(img, sigma_low, sigma_hi)
    # v2 = convolve(v, np.ones((5, 5, 5)) / 5**3, mode="same")
    pts = peak_local_max(v, min_distance=min_distance, threshold_abs=low_threshold)
    # vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]
    vals = v[pts[:, 0], pts[:, 1], pts[:, 2]]
    print(np.mean(vals))

    return pd.DataFrame({"z": pts[:, 0], "y": pts[:, 1], "x": pts[:, 2], "val": vals})


def load_tiffs_and_peaks(directory):
    tiff_files = natsorted([f for f in os.listdir(directory) if f.endswith('.tiff') or f.endswith('.tif')])
    csv_files = natsorted([f for f in os.listdir(directory) if f.endswith('.csv')])

    images = [tiff.imread(os.path.join(directory, f)) for f in tiff_files]
    peak_data = [pd.read_csv(os.path.join(directory, f)) for f in csv_files]

    return images, peak_data


class GuiProcess:

    def __init__(self, directory):

        self.directory = directory
        self.current_frame = 0
        self.image_data, self.peak_data_list = load_tiffs_and_peaks(directory)

        self.viewer = napari.Viewer(ndisplay=3)
        self.image_layer = None
        self.scatter_layer = None

        self.slider = ThresholdSlider(self.update_peaks, self.recompute_peaks)
        self.viewer.window.add_dock_widget(self.slider, area='right')

        # Bind keys for manual frame navigation
        self.viewer.bind_key("Right", lambda viewer: self.change_frame(1))
        self.viewer.bind_key("Left", lambda viewer: self.change_frame(-1))

    def change_frame(self, step):
        new_frame = self.current_frame + step

        # Ensure frame index stays within bounds
        if 0 <= new_frame < len(self.image_data):
            self.current_frame = new_frame
            self.update_frame()

    def update_frame(self):
        """Manually update frame display"""
        self.image_layer.data = np.swapaxes(self.image_data[self.current_frame], 0, 2)
        self.scatter_layer.data = self.peak_data_list[self.current_frame][['x', 'y', 'z']].values

    def run(self):

        self.image_layer = self.viewer.add_image(self.image_data[0], name='3D Image')
        self.scatter_layer = self.viewer.add_points(self.peak_data_list[0][['x', 'y', 'z']].values,
                                                    name='Peaks', size=10, face_color='red', out_of_slice_display=True)

        napari.run()

    def update_peaks(self, threshold):
        filtered_peaks = [data[data['val'] > threshold][['x', 'y', 'z']].values for data in self.peak_data_list]
        self.scatter_layer.data = filtered_peaks[self.current_frame]

    def recompute_peaks(self, sigma_high, sigma_low, min_distance):
        img = self.image_data[self.current_frame]
        peaks = get_peaks(img, sigma_high, sigma_low, min_distance=min_distance)
        self.peak_data_list[self.current_frame] = peaks
        self.update_frame()


def main():

    directory = r"D:\Tracking\NucleiTracking\data\interim\lightsheet\2025_02_06\recon\test"

    gui_process = GuiProcess(directory)
    gui_process.run()


if __name__ == "__main__":
    main()
