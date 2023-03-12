import os
import numpy as np
from NiftyHandler import NiftyHandler
from scipy.ndimage import label, binary_closing, binary_opening, \
    generate_binary_structure
from SpineSegmentation import SpineSegmentation


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
        denoised_ct = self.__denoise(thresholded_ct)
        body_segmentation = self.__compute_largest_connected_component(denoised_ct)
        return body_segmentation

    def isolate_bs(self, body_segmentation: np.ndarray) -> tuple[int,int]:
        """
        Given a clean segmentation of the entire body, return
        integers of the BB and CC slices. BB being the slice inferior
        slice of the lungs (lowest slice) and CC being the widest slice
        of the lungs. CC can be found where the lungs don't change much
        for the first time
        @param body_segmentation:
        @return:
        """

        # idea:
        lungs_segmentation = self.__isolate_lungs(body_segmentation)
        bb, cc = self.__find_bb_and_cc(lungs_segmentation)
        print(f"bb: {bb} cc: {cc}")
        return bb, cc


    def __isolate_lungs(self, body_segmentation: np.ndarray) -> np.ndarray:
        """
        given body segmentation, invert and find largest hole components being the lungs by
        inverting the image, finding components that are the holes and returning the holes
        @param body_segmentation:
        @return:
        """

        inverted_img = np.logical_not(body_segmentation)

        # slightly open the scan so it doesn't connect to the air outside the body
        rank = 3
        connectivity = 3
        structuring_element = generate_binary_structure(rank, connectivity)
        x = inverted_img
        inverted_img = binary_opening(x, structuring_element, iterations=3)
        labeled_img, num_features = label(inverted_img)
        sizes = np.bincount(labeled_img.ravel())


        sorted_cc = np.argsort(sizes)[::-1]
        k = 3
        # lungs should be the top 3 and 4 because of the body and outside of the body
        lung_labels = sorted_cc[2:4]
        lung1 = (labeled_img == lung_labels[0]).astype(int)
        lung2 = (labeled_img == lung_labels[1]).astype(int)
        return (lung1 | lung2).astype(float)



    def three_d_band(self, body_segmentation: np.ndarray, bb: int, cc: int) -> np.ndarray:
        """
        Given a body segmentation return segmentation of the body from bb to cc
        @param body_segmentation:
        @param bb:
        @param cc:
        @return:
        """

        body_band = np.zeros(body_segmentation.shape)
        body_band[:, :, bb:cc] = body_segmentation[:, :, bb: cc]
        return body_band

    def __denoise(self, thresholded_ct: np.ndarray) -> np.ndarray:
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

    def __find_bb_and_cc(self, lungs_segmentation) -> tuple[int, int]:
        """
        given a segmentation of the lungs find bb slice (lowest slice in which a lung voxel exists
        and cc (first slice in which the lung area doesn't change much)
        @param lungs_segmentation:
        @return:
        """

        # finding BB, in numpy top of lungs is the highest slice so BB is argmax
        bb = np.argmax(np.any(lungs_segmentation, axis=(0, 1)))

        # find CC, start looking from BB, take first area in which the lung capacity doesn't change much
        slices_area = np.sum(lungs_segmentation[:, :, bb:], axis=(0, 1))
        diffs = np.abs(np.diff(slices_area))
        cc = np.where(diffs <= 50)[0][0] + bb

        return bb, cc


class MergedROI:
    @staticmethod
    def merged_roi(ct_scan_path: str, anchor_segmentation_path: str, anchor_coordintes: tuple[int]):
        """
        global entry function for part 1 of the exercise
        @param ct_scan_path:
        @return:
        """

        output_dir = "3d_band_output"
        scan_basename = os.path.basename(ct_scan_path).split('.')[0]
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        data, orientation = NiftyHandler.read(ct_scan_path)
        dss = DifficultBodySegmentation()
        body_segmentation = dss.isolate_body(data)
        body_segmentation_path = os.path.join(output_dir, f"{scan_basename}_body_segmentation.nii.gz")
        NiftyHandler.write(body_segmentation, body_segmentation_path, orientation)

        bb, cc = dss.isolate_bs(body_segmentation)
        body_lung_band = dss.three_d_band(body_segmentation, bb, cc)
        lungs_segmentation_path = os.path.join(output_dir, f"{scan_basename}_lungs_segmentation.nii.gz")
        NiftyHandler.write(body_lung_band, lungs_segmentation_path, orientation)

        spine_segmentation = SpineSegmentation(ct_scan_path, anchor_segmentation_path, anchor_coordintes).getSpineSegmentation()




def main():

    ct_scan_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
    merged_roi(ct_scan_path)


if __name__ == '__main__':
    main()