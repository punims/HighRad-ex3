import os

import numpy as np
from NiftyHandler import NiftyHandler
from scipy.ndimage import label, binary_erosion, binary_closing, binary_dilation, binary_opening, generate_binary_structure

class DifficultBodySegmentation:
    """
    Class responsible for a more difficult segmentation of parts of the skeleton
    """

    def isolate_body(self, ct_scan: np.ndarray) -> np.ndarray:
        """
        Isolates the patient's body from the bed and the surroundings
        1. Does a basic thresholding.
        2. Filters out noise
        3. Computes the largest connected component.
        @param ct_scan: np array of the ct scan
        @return: returns segmentation of the body only
        """

        # First we threshold
        MIN_VAL = -500
        MAX_VAL = 2000
        thresholded_ct = ((ct_scan >= MIN_VAL) & (ct_scan <= MAX_VAL)).astype(float)
        denoised_ct = self.__denoise_body(thresholded_ct)
        body_segmentation = self.__compute_largest_connected_component(denoised_ct)
        return body_segmentation

    def isolate_bs(self):
        pass

    def three_d_band(self):
        pass

    def __denoise_body(self, thresholded_ct: np.ndarray) -> np.ndarray:
        """
        Given a thresholded scan of the body, denoise to get rid of
        artifacts.
        @param thresholded_ct:
        @return: cleaned version of the body
        """

        rank = 3
        connectivity = 2
        structuring_element = generate_binary_structure(rank, connectivity)
        x = thresholded_ct
        x = binary_opening(x, structuring_element, iterations=2)
        x = binary_closing(x, structuring_element)
        return x


    def __compute_largest_connected_component(self, denoised_ct):
        """
        given denoised ct return segmentation of the largest connected compoennt
        which is likely to be the body
        @param denoised_ct:
        @return:
        """

        x = denoised_ct
        x, num_components = label(x)
        x = x == np.argmax(np.bincount(x.flat)[1:]) + 1
        x = x.astype(float)
        return x

def get_skeleton_segmentation(ct_scan_path):
    """
    global entry function for part 1 of the exercise
    @param ct_scan_path:
    @return:
    """

    output_dir = "skeleton_segmentation_output"
    scan_basename = os.path.basename(ct_scan_path).split('.')[0]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    data, orientation = NiftyHandler.read(ct_scan_path)
    dss = DifficultBodySegmentation()
    body_segmentation = dss.isolate_body(data)
    body_segmentation_path = os.path.join(output_dir, f"{scan_basename}_body_segmentation.nii.gz")
    NiftyHandler.write(body_segmentation, body_segmentation_path, orientation)


def main():

    ct_scan_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
    get_skeleton_segmentation(ct_scan_path)


if __name__ == '__main__':
    main()