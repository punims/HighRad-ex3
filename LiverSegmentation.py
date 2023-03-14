import numpy as np
from NiftyHandler import NiftyHandler
from LiverROI import LiverROI
from MSRG import MSRG
from HardROI import DifficultBodySegmentation


class LiverSegmentation:
    """
    Class responsible for segmenting the liver
    """

    def __init__(self):
        self.liver_coordinates = (-25, 25, -30, 30, -10, 10)


    def segment(self, ct_scan_path: str, aorta_scan_path: str) -> None:
        """
        Segment the liver given a ct and a segmentation of the aorta.
        Saves it to the final file
        Parameters
        ----------
        ct_scan_path
        aorta_scan_path

        Returns
        -------



        """


        # get top boundary
        ct_data, orientation = NiftyHandler.read(ct_scan_path)
        body_segmentor = DifficultBodySegmentation(orientation)
        body_segmentation = body_segmentor.isolate_body(ct_data)
        _, upper_slice, _ = body_segmentor.isolate_bs(body_segmentation)

        # get liver roi seed
        liver_roi = LiverROI(self.liver_coordinates, "1")
        liver_segmentation_seed = liver_roi.get_liver_roi(ct_scan_path, aorta_scan_path)

        # get lower boundary using aorta.
        aorta_data, _ = NiftyHandler.read(aorta_scan_path)
        lower_slice = np.argmax(np.any(aorta_data, axis=(0, 1)))

        msrg = MSRG(axial_boundary=(lower_slice, upper_slice))
        liver_segmentation = msrg.run(ct_data, liver_segmentation_seed)
        NiftyHandler.write(liver_segmentation, "liver_segmentation_final", orientation)



if __name__ == '__main__':
    ct_scan_path = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
    aorta_nifti = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_Aorta.nii.gz"
    liver_segmentor = LiverSegmentation()
    liver_segmentor.segment(ct_scan_path, aorta_nifti)