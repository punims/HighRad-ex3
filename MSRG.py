from typing import Any
import numpy as np
from numpy import ndarray
from scipy.stats import norm
from NiftyHandler import NiftyHandler
from scipy.ndimage import generate_binary_structure, binary_dilation, binary_closing, binary_opening, \
    label


class MSRG:
    """
    Class implementing the MSRG (multiple seeded region growing) algorithm
    """

    def __init__(self, seed_number: int = 200, axial_boundary: tuple = None):
        self.seed_number = seed_number
        self.axial_boundary = axial_boundary
        self.__legal_methods = {"msrg", "msrg_roi"}

    def run(self, ct_data: np.ndarray, roi: np.array, method: str = "msrg_roi", threshold: float = 20,
            iterations: int = 130) -> tuple[
        ndarray, Any]:
        """
        Runs either one of the following methods:
        msrg:
        1. finds ~200 seeds within the ROI
        2. runs SRG algorithm for each seed
        3. applies morphological operations at the end

        msrg_roi:
        1. takes in entire roi as a giant seed
        2. grows the roi all at once
        3. applies morphological operations at the end.
        Parameters
        ----------
        threshold: threshold used in msrg to decide if to add a neighbor
        method: string to decide the algorithm
        roi: segmentation of the roi, maps directly to ct_data and has the same shape.
        ct_data : ct scan data.

        Returns
        -------

        """

        # slice in axial planes to reduce runtime slightly.
        original_shape = ct_data.shape
        if self.axial_boundary is not None:
            bottom, top = self.axial_boundary
            ct_data = ct_data[:, :, bottom:top]
            roi = roi[:, :, bottom:top]

        if method == "msrg":
            seeds = self.__generate_seeds(roi)
            segmentation = self.__msrg(ct_data, seeds, threshold)
        elif method == "msrg_roi":
            segmentation = self.__msrg_with_roi(ct_data, roi, threshold, iterations)
        else:
            raise ValueError(f"Trying to use illegal method type, legal method types are: {self.__legal_methods}")

        # return to normal
        if self.axial_boundary is not None:
            temp_segmentation = np.zeros(original_shape)
            bottom, top = self.axial_boundary
            temp_segmentation[:, :, bottom:top] = segmentation
            segmentation = temp_segmentation

        refined_segmentation = self.post_process(segmentation)
        return segmentation.astype(float), refined_segmentation.astype(float)

    def __generate_seeds(self, roi: np.ndarray) -> np.ndarray:
        """
        Given a roi, uniformly select ~200 indices from the roi.
        Parameters
        ----------
        roi: np array of segmentation that maps onto the ct scan data (has the same shape)

        Returns: subgroup of indices containing the roi
        -------

        """
        segmentation_indices = np.argwhere(roi == 1)
        assert segmentation_indices.shape[0] >= self.seed_number
        return np.random.permutation(segmentation_indices)[:self.seed_number]

    def __msrg(self, ct_data: np.ndarray, seeds: np.ndarray, threshold: float) -> np.ndarray:
        """
        Given ct_data and a self.seed_number x 3 array for seeds, return segmentation of the ct_data
        using msrg algorithm:
        1. for each seed run SRG
        2. combine srg outputs as a segmentation and return
        Parameters
        ----------
        ct_data
        seeds

        Returns
        -------

        """
        segmentation = np.zeros_like(ct_data, dtype=np.uint8)

        # Initialize list of active seeds
        active_seeds = list(seeds)

        # Loop until no more active seeds
        while len(active_seeds) > 0:
            # Pop a seed point from the list
            current_seed = active_seeds.pop(0)

            # Check if seed point is already segmented
            if segmentation[current_seed] == 1:
                continue

            # Initialize empty set for pixels to be segmented
            to_segment = {current_seed}

            # Loop until no more pixels to be segmented
            while len(to_segment) > 0:
                # Pop a pixel from the set
                current_pixel = to_segment.pop()

                # Check if pixel is within image bounds
                x, y, z = current_pixel
                if x < 0 or x >= ct_data.shape[0] or y < 0 or y >= ct_data.shape[1] or z < 0 or z >= ct_data.shape[2]:
                    continue

                # Check if pixel is already segmented
                if segmentation[current_pixel] == 1:
                    continue

                # Check if pixel intensity is within threshold of seed intensity
                if abs(ct_data[current_pixel] - ct_data[current_seed]) > threshold:
                    continue

                # Segment pixel
                segmentation[current_pixel] = 1

                # Add unsegmented neighbors to set of pixels to be segmented
                neighbors = [(x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)]
                to_segment.update([n for n in neighbors if segmentation[n] == 0])

            # Add all segmented pixels to list of active seeds for next iteration
            active_seeds.extend(list(to_segment))

        return segmentation

    def __msrg_with_roi(self, ct_scan: np.ndarray, roi: np.ndarray, threshold: float = 10, iterations: int = 130):
        """
        Assuming the entire ROI is nearly contained within the confines of the liver
        1. preprocess the ROI a bit to get rid of outlier voxels
        2. use region growing with the entire ROI
        3. return final segmentation
        Parameters
        ----------
        ct_scan: np array of the ct scan data
        roi: segmentation of the ROI in the liver, assumes most if not all the roi is inside the liver

        Returns
        -------

        """
        # refine the ROI
        roi = roi.astype(bool)
        x_shape = np.max(np.sum(roi, axis=0))
        y_shape = np.max(np.sum(roi, axis=1))
        z_shape = np.max(np.sum(roi, axis=2))

        segmentation = np.zeros(roi.shape)
        segmentation[roi] = MSRG.sample_roi(ct_scan[roi].reshape(x_shape, y_shape, z_shape)).flatten()
        segmentation = segmentation > 0


        # Run MSRG with all seeds at once using dilation. Do so until convergence or after x iterations
        prev_size = np.sum(segmentation)
        cur_size = 0
        i = 0
        structure_element = generate_binary_structure(3, 2)
        while np.abs(prev_size - cur_size) > 100 and i < iterations:
            # get neighbors using dilation and bitwise xor
            prev_size = np.sum(segmentation)
            neighbors = np.bitwise_xor(segmentation, binary_dilation(segmentation, structure_element))

            # find which neighbors are close to the current mean and add them
            cur_average = np.mean(ct_scan[segmentation])
            neighbors_to_add = np.logical_and(neighbors, np.abs(ct_scan - cur_average) <= threshold)
            segmentation = np.logical_or(neighbors_to_add, segmentation)
            cur_size = np.sum(segmentation)
            i += 1

        return segmentation

    @staticmethod
    def sample_roi(roi: np.ndarray, sample_size: int = 1000, bottom_percentile: int = 25, top_percentile: int = 75) -> np.ndarray:
        """
        Given an ROI cube from within the CT, sample it by taking 'sample_size' points with a gaussian distribution form the middle.
        Only take values between the bottom and top percentiles.
        Parameters
        ----------
        roi: CT scan in the ROI
        sample_size: amount of voxels that should be a part of the end segmentation
        bottom_percentile: bottom percentile of the distribution of values
        top_percentile: top percentile of the distribution of values

        Returns sampled ROI with the same shape as the original
        -------

        """

        # Get the values between the bottom and top percentiles
        bottom_value = np.percentile(roi, bottom_percentile)
        top_value = np.percentile(roi, top_percentile)
        roi_thresholded = (roi >= bottom_value) & (roi <= top_value)

        # Create a binary array with the same shape as the input array
        binary_array = np.zeros(roi.shape, dtype=bool)
        std = [np.std(list(range(binary_array.shape[0]))), np.std(list(range(binary_array.shape[1]))), np.std(list(range(binary_array.shape[2])))]


        # Sample voxels from a gaussian distribution centered in the middle of the array
        middle = np.array(roi.shape) // 2
        for i in range(sample_size):
            voxel = None
            while voxel is None or binary_array[voxel[0], voxel[1], voxel[2]] or not roi_thresholded[voxel[0], voxel[1], voxel[2]]:
                # Sample a voxel from the gaussian distribution
                voxel = np.round(norm.rvs(size=3, loc=middle, scale=std)).astype(int)
                # Make sure the voxel is within the bounds of the array
                voxel = np.clip(voxel, 0, np.array(roi.shape) - 1)[: ,np.newaxis]

            # Set the voxel to 1 in the binary array
            binary_array[voxel[0], voxel[1], voxel[2]] = True

        return binary_array

    def post_process(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Given a rough segmentation of the liver use morphological post processing to get a refined version
        Parameters
        ----------
        segmentation: rough segmentation of the liver

        Returns refined segmentation
        -------

        """
        # use binary closing and opening
        small_element = generate_binary_structure(3, 1)
        structure_element = generate_binary_structure(3, 2)
        x = segmentation
        x = binary_closing(x, small_element, iterations=3)
        x = binary_opening(x, structure_element, iterations=1)

        # find largest connected component and return it.
        x, num_components = label(x)
        x = x == np.argmax(np.bincount(x.flat)[1:]) + 1
        x = x.astype(float)

        return x


def script_run():
    """
    enclosing method to run the script
    can change refine to just run refine stage or entire msrg method.
    Returns
    -------

    """

    seed_size = 200
    msrg = MSRG(seed_size, (125, 290))
    liver_segmentation_path = "liver_segmentation_1"
    refined_liver_segmentation_path = "liver_segmentation_1_refined"
    refine = False

    if refine:
        liver_segmentation_data, liver_orientation = NiftyHandler.read(liver_segmentation_path)
        refined_liver = msrg.post_process(liver_segmentation_data)
        NiftyHandler.write(refined_liver, refined_liver_segmentation_path, liver_orientation)
    else:
        ct_scan_path = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
        roi_path = f"/home/edan/Desktop/HighRad/Exercises/ex3/liver_roi_output/liver_roi_1.nii.gz"
        ct_data, ct_orientation = NiftyHandler.read(ct_scan_path)
        roi_data, roi_orientation = NiftyHandler.read(roi_path)
        unrefined_segmentation, final_segmentation = msrg.run(ct_data, roi_data, threshold=10, iterations=110)
        NiftyHandler.write(final_segmentation, refined_liver_segmentation_path, ct_orientation)
        NiftyHandler.write(unrefined_segmentation, liver_segmentation_path, ct_orientation)


if __name__ == '__main__':
    script_run()
