from os import environ as env
from os import path
from os import mkdir
from typing import Tuple
from NiftyHandler import NiftyHandler
import numpy as np
from scipy.ndimage import label, binary_erosion, binary_closing, binary_dilation, generate_binary_structure
from matplotlib.pyplot import plot, show, xlabel, ylabel, title
import multiprocessing as mp


class SkeletonSegmentation:
    """
    Class responsible for segmenting bone in CT scans provided by nifty files.
    The segmentation is done by thresholding voxel values.
    This class is also responsible for finding a min thershold for the segmentation and once found does post
    processing operations in the method SkeletonTHFinder
    """

    MAX_TH = 1300

    def __init__(self, nifti_file_path):
        self.__nifti_file_path = nifti_file_path
        self.__output_folder = "skeleton_output"
        if not path.isdir(self.__output_folder):
            mkdir(self.__output_folder)
        self.__output_nifti_basename = path.basename(nifti_file_path)

    def segmentationByTH(self, data: np.ndarray, imin: float, imax: float = MAX_TH, orientation: np.ndarray = None) -> Tuple[int, np.ndarray]:
        """

        :param data: data from nifti file as an ndarray
        :param imin: minimal voxel value to be considered as a bone voxel
        :param imax: max voxel value to be considered as a bone voxel
        :return: 1 if successful, 0 otherwise. also returns segmented data.
        """

        SUCCESS = 1
        if data is None:
            raise TypeError("Data was corrupt, check your file.")
        boolean_segmented_data = (imax >= data) & (data >= imin)
        binary_segmented_data = boolean_segmented_data.astype(float)
        output_file_name = f"{path.join(self.__output_folder, self.__output_nifti_basename)}_seg_{imin}_{imax}"
        NiftyHandler.write(binary_segmented_data, output_file_name, orientation)

        return SUCCESS, binary_segmented_data

    """
    class responsible for finding a minimal threshold for SkeletonSegmentation
    """

    def skeletonTHFinder(self):
        """
        This function iterates over 25 candidate Imin thresholds in the range of [150,500] (with
        intervals of 14). In each run, use the SegmentationByTH function you’ve implemented, and
        count the number of connectivity components in the resulting segmentation
        Choose the imin which is the first or second minima
        performs post-processing (morphological operations – cleans out single
        pixels, closes holes, etc.) until left with a single connectivity component.
        Finally saves the segmentation file.

        :param nifti_file:
        :return:
        """

        data, orientation = NiftyHandler.read(self.__nifti_file_path)
        connected_components = self.__multiprocess_skeletonTHFinder(data, orientation)
        # connected_components = self.__no_multiprocess_skeletonTHFinder(data)
        self.__plot_connected_components(connected_components)
        best_segmentation = min(connected_components, key=lambda x: x[1])
        processed = self.__post_processing(best_segmentation)
        self.__save_processed_segmentation(processed, orientation)


    def skeletonSegmentation(self, imin_value: int) -> np.ndarray:
        """
        Same functionality as skeletonTHFinder except that this method is given the minimal ivalue instead
        of searching for it and this method also returns the skeleton segmentation itself.
        Parameters
        ----------
        imin_value: minimal ivalue for initial thresholding

        Returns: ndarray of the skeleton segmentation
        -------

        """

        data, orientation = NiftyHandler.read(self.__nifti_file_path)
        _, thresholded_data = self.segmentationByTH(data, imin_value, orientation=orientation)
        processed = self.__post_processing((imin_value, 1))
        self.__save_processed_segmentation(processed, orientation)
        return processed

    def morphological_postprocessing(self, segmentation_imin: int, connected_components: int = 0) -> None:
        """
        only do morphological post processing on a nifti file
        """

        processed = self.__post_processing((segmentation_imin, connected_components))
        self.__save_processed_segmentation(processed)

    def __post_processing(self, best_segmentation: Tuple[int, int]) -> np.ndarray:
        """
        Does post processing on the best segmentation. The post processing fills in single voxels and
        connected all connected components until only one connected component exists.
        :return:
        """
        RANK = 3
        CONNECTIVITY = 1
        structuring_element = generate_binary_structure(RANK, CONNECTIVITY)
        segmentation_imin = best_segmentation[0]
        if env.get("DEBUG"):
            print(f"Best segmentation had an imin value of {segmentation_imin} ")
        x, _ = NiftyHandler.read(
            f"{path.join(self.__output_folder, self.__output_nifti_basename)}_seg_{segmentation_imin}_1300")
        # We'll simply do closing to fill gaps and then opening to get rid of outliers.
        x = binary_dilation(x, structuring_element)
        x = binary_closing(x, structuring_element)
        x = binary_erosion(x, structuring_element)
        x = binary_dilation(x, structuring_element)

        # get labels and only take largest connected component as the skeleton and get rid of large outliers.
        x, num_components = label(x)
        if env.get("DEBUG"):
            print(f"the processed segmentation has {num_components} connected components vs {best_segmentation[1]} original connected components")
        x = x == np.argmax(np.bincount(x.flat)[1:]) + 1
        x = x.astype(float)
        processed_connected_components = self.__count_connected_components(x)
        if env.get("DEBUG"):
            print(f"end result has {processed_connected_components} connected components vs {best_segmentation[1]} original connected components")
        return x

    def __save_processed_segmentation(self, processed: np.ndarray, orientation: np.ndarray = None) -> None:
        """
        saves the processed image segmentation
        """

        segmentation_path = f"{path.join(self.__output_folder, self.__output_nifti_basename)}_SkeletonSegmentation.nii.gz"
        NiftyHandler.write(processed, segmentation_path, orientation)

    def __no_multiprocess_skeletonTHFinder(self, data: np.ndarray, orientation: np.ndarray = None) -> list[tuple[int, int]]:
        MIN_ITER_VAL = 150
        MAX_ITER_VAL = 500
        STEP = 14
        ret_arr = []
        for min_val in range(MIN_ITER_VAL, MAX_ITER_VAL, STEP):
            result, segmentation = self.segmentationByTH(data, min_val, orientation=orientation)
            if result:
                # save corresponding imin and number of connected components in the same list
                connected_components = self.__count_connected_components(segmentation)
                ret_arr.append((min_val, connected_components))
        return ret_arr

    def __multiprocess_skeletonTHFinder(self, data: np.ndarray, orientation: np.ndarray = None, num_processes: int = 4) -> []:
        """
        Multiprocess version of the skeletonTHFinder.
        :param data:
        :param num_processes:
        :return:
        """
        MIN_ITER_VAL = 150
        MAX_ITER_VAL = 500
        STEP = 14
        manager = mp.Manager()
        lock = manager.Lock()
        my_range = range(MIN_ITER_VAL, MAX_ITER_VAL, STEP)
        args = [(data, min_val, lock, orientation) for shared_array, min_val, lock, orientation in
                zip([data] * len(my_range), my_range, [lock] * len(my_range), [orientation]*len(my_range))]
        queue = mp.Queue()
        pool = mp.Pool(num_processes)

        # Call the function repeatedly in parallel using the multiprocessing pool
        if env.get('DEBUG'):
            print("Starting multiprocess segmentation")
        for arg in args:
            pool.apply_async(self._skeletonTHFinder_task, args=arg, callback=queue.put)
        pool.close()
        pool.join()
        if env.get("DEBUG"):
            print("Finished multiprocess segmentation")
        connected_components = []
        for _ in range(len(args)):
            connected_components.append(queue.get())
        connected_components.sort(key=lambda x: x[0])
        return connected_components

    def _skeletonTHFinder_task(self, data: np.ndarray, min_val: int, lock: mp.Lock, orientation: np.ndarray = None) -> Tuple[int, int]:
        """
        protected because we cant send private methods to apply_async because of name mangling.
        Does the task of segmentation according to the min val and returns a tuple of the min val
        and the number of connected components
        :param data:
        :param min_val:
        :return:
        """
        try:
            # Need lock so as not to access the numpy array simultaneously.
            lock.acquire()
            result, segmentation = self.segmentationByTH(data, min_val, orientation=orientation)
        finally:
            lock.release()
        if result:
            # save corresponding imin and number of connected components in the same list
            connected_components = self.__count_connected_components(segmentation)
            return (min_val, connected_components)

    def __count_connected_components(self, matrix: np.ndarray) -> int:
        # Apply connected component labeling to the binary matrix
        labeled_matrix, num_components = label(matrix)
        return num_components

    def __plot_connected_components(self, data: np.ndarray) -> None:
        x = [t[0] for t in data]
        y = [t[1] for t in data]
        plot(x, y)
        xlabel('imin')
        ylabel('connected components')
        title('imin vs connected components')
        show()


if __name__ == '__main__':

    """
    Enter absolute path to the CT scan in ct_path and run. 
    """
    ct_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
    imin_val = 248
    SkeletonSegmentation(ct_path).skeletonSegmentation(imin_val)
