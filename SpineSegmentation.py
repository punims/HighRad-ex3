import numpy as np
from NiftyHandler import NiftyHandler
from SkeletonSegmentation import SkeletonSegmentation
from scipy.ndimage import label, generate_binary_structure, binary_closing


class SpineSegmentation:
    """
    Class responsible for segmenting the spine from a CT scan
    """
    def __init__(self, ct_nifti: str, anchor_segmentation_nifti: str, anchor_coordinates: tuple[int]):
        """
        Constructor
        Parameters
        ----------
        anchor_coordinates: x_left, x_right, y_up, y_down coordinates to focus in on
        ct_nifti: path to ct
        anchor_segmentation_nifti: path to anchor

        """
        self.anchor_coordinates = anchor_coordinates
        self.anchor_segmentation_nifti = anchor_segmentation_nifti
        self.ct_nifti = ct_nifti
        self.skeletonSegmentor = SkeletonSegmentation(self.ct_nifti)

    def getSpineSegmentation(self) -> np.ndarray:
        """
        Given a ct scan and a some anchor segmentation gets a segmentation
        of the spine

        Returns: Segmentation of the spine
        -------

        """
        anchor_data, anchor_orientation = NiftyHandler.read(self.anchor_segmentation_nifti)
        imin_value = 248
        skeleton_data = self.skeletonSegmentor.skeletonSegmentation(imin_value)
        roi = self.__spine_crop(skeleton_data, anchor_data)
        clean_spine = self.__refine_spine(roi)
        return clean_spine


    def __spine_crop(self, skeleton_data: np.ndarray, anchor_data: np.ndarray):
        """
        Given the skeleton segmentation, use the anchor data and tuple to get a better ROI of only the spine itself.
        We use the center of mass of the anchor and positional knowledge of it to get the spine

        Parameters
        ----------
        skeleton_data: binary segmentation of the skeleton
        anchor_data: some anchor segmentation

        Returns
        -------

        """

        # assume anchor is the aorta and for every slice delete skeleton segmentation that is in front of the aorta.
        segmentation_indices = np.where(anchor_data > 0)
        x = np.max(segmentation_indices[0])
        y = np.min(segmentation_indices[1])

        x_left, x_right, y_up, y_down = self.anchor_coordinates
        spine_crop = np.zeros(skeleton_data.shape)
        spine_crop[x+x_left: x+x_right, y+y_up:y+y_down, :] = skeleton_data[x+x_left: x+x_right, y+y_up:y+y_down, :]
        return spine_crop

    def __refine_spine(self, spine_data: np.ndarray) -> np.ndarray:
        """
        Given a rough spine crop do the following:
        1) Only work on largest connected component being only the spine
        2) Try opening to get rid of ribs?
        3) Possible smoothing using median pass?
        Parameters
        ----------
        spine_data: ndarray rough segmentation of the spine

        Returns: cleaner ndarray of the segmentation of the spine
        -------

        """

        x, num_components = label(spine_data)
        largest_component = x == np.argmax(np.bincount(x.flat)[1:]) + 1  # gets the spine
        medium_structuring_element = generate_binary_structure(3, 2)
        closed_holes = binary_closing(largest_component, medium_structuring_element, iterations=2)

        return (closed_holes > 0).astype(float)









if __name__ == '__main__':

    i = 4
    ct_nifti = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case{i}_CT.nii.gz"
    aorta_nifti = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case{i}_Aorta.nii.gz"
    coordinates = (-80, 50, -70, 100)
    spine_segmentation = SpineSegmentation(ct_nifti, aorta_nifti, coordinates)
    spine_segmentation.getSpineSegmentation()