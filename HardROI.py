import numpy as np


class DifficultSkeletonSegmentation:
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
        MIN_VAL = 500
        MAX_VAL = 2000
        thresholded_ct = ((ct_scan >= MIN_VAL) & (ct_scan <= MAX_VAL)).astype(int)

        denoised_ct = self.__denoise_body(thresholded_ct)
        body_segmentation = self.__compute_largest_connected_component(denoised_ct)

        return body_segmentation

    def isolate_bs(self):
        pass

    def three_d_band(self):
        pass

    def __denoise_body(self, thresholded_ct):
        pass

    def __compute_largest_connected_component(self, denoised_ct):
        pass

