import numpy as np


class SegmentationMetricScorer:

    @staticmethod
    def score_selection(scores: list[str], segmentation: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
        """
        This method takes in a list of scores to return
        Currently this method supports returning dice and vod scores.

        If a score that is passed is illegal, raises a ValueError
        otherwise returns a dictionary of the score name mapped to the score.
        Parameters
        ----------
        segmentation: np.ndarray of the segmentation ground truth
        prediction: np.ndarray of the prediction
        scores: names of the scores to return

        Returns
        -------

        returns a dictionary of the score name mapped to the score.

        """
        legal_scores = {
                "dice": SegmentationMetricScorer.dice_score,
                "vod": SegmentationMetricScorer.vod_score,
            }

        scores_dict = dict()

        for score in scores:
            if score not in legal_scores.keys():
                raise ValueError(f"score of type {score} is not supported. Legal score types are {legal_scores}")
            else:
                scores_dict[score] = legal_scores[score](segmentation, prediction)

        return scores_dict

    @staticmethod
    def dice_score(segmentation: np.ndarray, prediction: np.ndarray) -> float:
        """
        Given data from a segmentation file and predicted data
        calculates the dice score between the two segmentations

        Assumes both segmentations have the same shape otherwise the method
        raises a ValueError
        """

        if segmentation.shape != prediction.shape:
            raise ValueError("Segmentation and Prediction have different shapes")

        segmentation = (segmentation > 0)
        prediction = (prediction > 0)
        overlap = np.sum(segmentation & prediction)
        total_area = np.sum(segmentation) + np.sum(prediction)
        return 2 * overlap / total_area

    @staticmethod
    def vod_score(segmentation: np.ndarray, prediction: np.ndarray) -> float:
        """
        Given path to a segmentation file and predicted path file
        calculates the vod score between the two segmentations
        Volume Overlap Differnece == 1 - IOU (intersection over union)

        Assumes both segmentations have the same shape otherwise the method
        raises a ValueError
        """

        if segmentation.shape != prediction.shape:
            raise ValueError("Segmentation and Prediction have different shapes")

        segmentation = segmentation > 0
        prediction = prediction > 0
        intersection = np.sum(segmentation & prediction)
        union = np.sum(segmentation | prediction)
        return 1 - (intersection / union)




