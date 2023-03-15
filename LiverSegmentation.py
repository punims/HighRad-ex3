import numpy as np
from NiftyHandler import NiftyHandler
from LiverROI import LiverROI
from MSRG import MSRG
from HardROI import DifficultBodySegmentation
from SegmentationMetricScorer import SegmentationMetricScorer


class LiverSegmentation:
    """
    Class responsible for segmenting the liver
    """

    def __init__(self):
        self.liver_coordinates = (-25, 25, -30, 30, -10, 10)

    def evaluate_segmentation(self, predicted_segmentation_path: str, ground_truth_path: str) -> None:
        """
        Given strings to the prediction and ground truth, reads them and prints the vod and dice scores
        Parameters
        ----------
        predicted_segmentation_path
        ground_truth_path

        Returns none
        -------

        """

        prediction_data, _ = NiftyHandler.read(predicted_segmentation_path)
        gt_data, _ = NiftyHandler.read(ground_truth_path)
        scores = SegmentationMetricScorer.score_selection(["dice", "vod"], gt_data, prediction_data)
        for k, v in scores.items():
            print(f"{k}:{v}")

    def segment(self, ct_scan_path: str, aorta_scan_path: str, output: str) -> None:
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
        _, _, upper_slice = body_segmentor.isolate_bs(body_segmentation)

        # get liver roi seed
        liver_roi = LiverROI(self.liver_coordinates, "1")
        liver_segmentation_seed = liver_roi.get_liver_roi(ct_scan_path, aorta_scan_path)

        # get lower boundary using aorta.
        aorta_data, _ = NiftyHandler.read(aorta_scan_path)
        lower_slice = np.argmax(np.any(aorta_data, axis=(0, 1)))

        msrg = MSRG(axial_boundary=(lower_slice, upper_slice))
        unrefined_segmentation, liver_segmentation = msrg.run(ct_data, liver_segmentation_seed)
        NiftyHandler.write(unrefined_segmentation, f"{output}_unrefined", orientation)
        NiftyHandler.write(liver_segmentation, output, orientation)


if __name__ == '__main__':
    """
    Run entire pipeline.
    Finds liver in the ct given the aorta as an anchor point
    Prints out the dice and vod scores compared to the gt segmentation.
    
    ct_scan_path: path to the full body ct
    aorta_nifti: path to the aorta nifti file
    gt: path to the gt of the liver
    output: name of the output file
    """
    ct_scan_path = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_CT.nii.gz"
    aorta_nifti = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_Aorta.nii.gz"
    gt = f"/home/edan/Desktop/HighRad/Exercises/data/Targil1_data-20230227T131201Z-001/Targil1_data/Case1_liver_segmentation.nii.gz"
    output = "final_liver_segmentation"
    liver_segmentor = LiverSegmentation()
    liver_segmentor.segment(ct_scan_path, aorta_nifti, output)
    liver_segmentor.evaluate_segmentation(output, gt)
