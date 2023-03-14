from typing import Any

import numpy as np
from os.path import isdir, join
from os import mkdir
from NiftyHandler import NiftyHandler



class LiverROI:
    """
    Class that is responsible for retrieval of the liver roi
    """

    IMIN = -100
    IMAX = 200

    def __init__(self,  coordinates: tuple[int], output_file_name: str):
        """
        Constructor
        coordinates: x_left, x_right, y_up, y_down, z_up, z_down coordinates for the roi
        Parameters
        ----------
        coordinates
        """

        self.coordinates = coordinates
        self.output = "liver_roi_output"
        self.output_file_name = output_file_name
        self.estimated_liver_coordinate = None
        if not isdir(self.output):
            mkdir(self.output)

    def get_liver_roi(self, ct_scan_path: str, aorta_segmentation_path: str) -> np.ndarray:
        """
        Given a path to a ct scan and the aorta segmentation, return the ROI of the liver.
        Parameters
        ----------
        ct_scan_path: path to the ct scan
        aorta_segmentation_path: path to the aorta segmentation
        path to the aorta segmentation
        Returns
        -------
        """


        ct_data, ct_orientation = NiftyHandler.read(ct_scan_path)
        aorta_data, aorta_orientation = NiftyHandler.read(aorta_segmentation_path)

        # find estimated liver coordinate using the aorta.
        estimated_liver_coord = self.__estimate_liver_midpoint(aorta_data)

        # use coordinates to get a rough estimate location ROI of the liver
        liver_segmentation, liver_roi = self.__crop_liver_roi(ct_data, estimated_liver_coord)

        # threshold using IMIN and IMAX
        liver_segmentation = ((LiverROI.IMAX >= liver_segmentation) & (liver_segmentation >= LiverROI.IMIN)).astype(float)

        liver_roi_path = join(self.output, f"liver_roi_{self.output_file_name}")
        NiftyHandler.write(liver_segmentation, liver_roi_path, ct_orientation)

        # return ROI segmenation
        return liver_segmentation

    def __estimate_liver_midpoint(self, aorta_data: np.ndarray, z_translation: int = 0) -> tuple[int | Any, Any, Any]:
        """
        return the estimated midpoint of the liver using the aorta
        as an anchor
        Returns
        -------

        """
        aorta_axial_indices = np.nonzero(np.any(aorta_data, axis=(0, 1)))[0]
        z_coord = aorta_axial_indices[len(aorta_axial_indices)//2] + z_translation
        average_aorta_cord = np.average(np.where(aorta_data > 0), axis=1)
        x_coord = average_aorta_cord[0] + 150
        y_coord = average_aorta_cord[1]

        return x_coord, y_coord, z_coord



    def __crop_liver_roi(self, ct_data, estimated_liver_middle_coordinates):
        """
        return cropped version of the ct scan
        relies on aorta_halfway coordinate for the z axis
        Parameters
        ----------
        estimated_liver_middle_coordinates

        Returns
        -------

        """

        x, y, z = estimated_liver_middle_coordinates
        x = int(x)
        y = int(y)
        z = int(z)
        x_left, x_right, y_in, y_out, z_up, z_down = self.coordinates
        liver_segmentation = np.zeros(ct_data.shape) - 1000
        liver_segmentation[x+x_left:x+x_right, y+y_in:y+y_out, z+z_up: z+z_down] = 1
        liver_crop = ct_data[x+x_left:x+x_right, y+y_in:y+y_out, z+z_up: z+z_down]
        return liver_segmentation, liver_crop



    def get_coordinates(self):
        """
        getter for coordinates
        Returns
        -------

        """
        return self.coordinates



if __name__ == '__main__':

    for i in range(1,5):
        print(f"Working on scan {i}")
        ct_scan_path = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case{i}_CT.nii.gz"
        aorta_nifti = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case{i}_Aorta.nii.gz"
        coordinates = (-25, 25, -30, 30, -10, 10)
        liver_roi = LiverROI(coordinates, i)
        roi = liver_roi.get_liver_roi(ct_scan_path, aorta_nifti)

