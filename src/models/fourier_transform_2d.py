from skimage.measure import regionprops, find_contours
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt


def parametrize_nuclei_from_image(label_image, **kwargs):
    """
    This function takes a label image and returns a dataframe of nuclei parameters
    """

    fourier_array = []
    inverse_fourier_array = []

    for prop in regionprops(label_image):
        angle = prop.orientation

        contour = find_contours(np.pad(prop.image, (1, 1)), 0.5)[0]
        centroid = prop.centroid_local

        contour = contour - centroid

        # Get the fourier transform
        fourier = fourier_transform_2d(contour, angle, **kwargs)

        # Get the inverse fourier transform
        inverse_fourier = inverse_fourier_transform_2d(fourier, angle, **kwargs)

        fourier_array.append(fourier)
        inverse_fourier_array.append(inverse_fourier)

    return fourier_array, inverse_fourier_array

# def plot_contours(prop, **kwargs):
#     """
#     # Plot the results
#     plt.figure()
#     plt.plot(contour[:, 0], contour[:, 1], "k")
#     plt.plot(inverse_fourier[:, 0], inverse_fourier[:, 1], "r")
#     plt.axis("equal")
#     plt.show()
#     """


def single_image_transform(prop, **kwargs):
    """
    This function takes a single image and returns the fourier transform of the polar coordinates
    """
    # Get the contour
    contour = find_contours(prop.image, 0.5)[0]

    # Get the centroid and orientation
    centroid = prop.centroid_local
    angle = prop.orientation

    # Get the fourier transform
    fourier = fourier_transform_2d(contour, angle, **kwargs)

    # Get the inverse fourier transform
    inverse_fourier = inverse_fourier_transform_2d(fourier, angle, **kwargs)

    return fourier


def fourier_transform_2d(contour, correction_angle=0, level=8):
    """
    This function takes a contour and returns the fourier transform of the polar coordinates
    """

    # Get the polar coordinates
    r = np.sqrt(contour[:, 0] ** 2 + contour[:, 1] ** 2)
    theta = np.arctan2(contour[:, 1], contour[:, 0])

    theta, r = correct_duplicates(theta, r)



    # Correct the angle
    theta = theta - correction_angle
    theta = theta % (2 * np.pi)

    # sort the polar coordinates
    sort_ind = np.argsort(theta)
    theta = theta[sort_ind]
    r = r[sort_ind]

    # add the first point to the end, and the last point to the beginning
    theta = np.append(theta, theta[0] + 2 * np.pi)
    r = np.append(r, r[0])
    theta = np.insert(theta, 0, theta[-2] - 2 * np.pi)
    r = np.insert(r, 0, r[-2])

    # Interpolate the polar coordinates
    r_interp = interpolate.interp1d(theta, r, kind="cubic")
    theta_interp = np.linspace(0, 2 * np.pi, 2 ** level)
    r_interp = r_interp(theta_interp)
    r_interp = r_interp/np.mean(r_interp)

    # Get the fourier transform
    fourier = np.fft.fft(r_interp)

    # print(np.min(r_interp), np.max(r_interp), fourier[0], np.mean(r_interp), fourier[0]/2**level, fourier[1])

    return fourier


def correct_duplicates(theta, r):
    """
    finds duplicate theta values and averages the r values
    """
    # Get the unique theta values
    unique_theta = np.unique(theta)

    # Get the average r values
    unique_r = np.zeros(len(unique_theta))
    for i, t in enumerate(unique_theta):
        unique_r[i] = np.mean(r[theta == t])

    return unique_theta, unique_r


def inverse_fourier_transform_2d(fourier, correction_angle=0, level=8):
    """
    This function takes a fourier transform and returns the inverse fourier transform
    """

    # Get the inverse fourier transform
    r_interp = np.fft.ifft(fourier)
    theta_interp = np.linspace(0, 2 * np.pi, 2 ** level)
    theta_interp = theta_interp + correction_angle

    # Get the cartesian coordinates
    x_interp = r_interp * np.cos(theta_interp)
    y_interp = r_interp * np.sin(theta_interp)

    return np.stack([x_interp, y_interp]).T



