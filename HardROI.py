import os
import numpy as np
from NiftyHandler import NiftyHandler
from scipy.ndimage import label, binary_closing, binary_opening, \
    generate_binary_structure, binary_erosion, binary_dilation, iterate_structure
from scipy.spatial import ConvexHull, Delaunay
from SpineSegmentation import SpineSegmentation


class DifficultBodySegmentation:
    """
    Class responsible for a more difficult segmentation of parts of the skeleton
    """

    def __init__(self, orientation):
        self.orientation = orientation

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

    def isolate_bs(self, body_segmentation: np.ndarray) -> tuple[np.ndarray, int, int]:
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
        return lungs_segmentation, bb, cc

    def __isolate_lungs(self, body_segmentation: np.ndarray) -> np.ndarray:
        """
        given body segmentation, invert and find largest hole components being the lungs by
        inverting the image, finding components that are the holes and returning the holes
        @param body_segmentation:
        @return:
        """

        # fill large holes (lungs) which gets us the full body
        structuring_element = iterate_structure(generate_binary_structure(3, 3), iterations=2)
        full_body = binary_dilation(body_segmentation, structuring_element, iterations=20)
        full_body = binary_erosion(full_body, structuring_element, iterations=20) > 0

        # with the full body we can get the air in the background
        air = ~full_body
        body_and_background_segmentation = (body_segmentation > 0) | air

        # invert to get all air in the body, largest connected components should be the lungs
        inverted_img = np.logical_not(body_and_background_segmentation)

        # slightly open the scan so that it doesn't connect to the lungs
        rank = 3
        connectivity = 3
        structuring_element = generate_binary_structure(rank, connectivity)
        x = inverted_img
        inverted_img = binary_opening(x, structuring_element, iterations=7)
        labeled_img, num_features = label(inverted_img)
        sizes = np.bincount(labeled_img.ravel())

        sorted_cc = np.argsort(sizes)[::-1]
        # lungs should be the top 2 and 3. 1 is always the entire image
        lung_labels = sorted_cc[1:3]
        lung1 = (labeled_img == lung_labels[0]).astype(int)
        lung2 = (labeled_img == lung_labels[1]).astype(int)
        return (lung1 | lung2).astype(float)

    def three_d_band(self, body_segmentation: np.ndarray, lungs_segmentation: np.ndarray, bb: int,
                     cc: int) -> np.ndarray:
        """
        Given a body segmentation return segmentation of area between the body and the convex hull of the lungs
        from bb to cc
        @param body_segmentation:
        @param bb:
        @param cc:
        @return:
        """

        # get filled convex hull of lungs and get whatever is between the body and the convex hull
        lungs_convex_hull, _ = self.__flood_fill_hull(lungs_segmentation)
        body_without_hull = (~(lungs_convex_hull > 0) & (body_segmentation > 0)).astype(float)

        # get the cut between bb and cc
        body_band = np.zeros(body_without_hull.shape)
        body_band[:, :, bb:cc] = body_without_hull[:, :, bb: cc]
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

    def __flood_fill_hull(self, image: np.ndarray) -> tuple[np.ndarray, ConvexHull]:
        """
        Gets convex hull of a binary image and fills it.
        Parameters
        ----------
        image: binary image

        Returns: filled convex hull and the hull object
        -------

        """
        points = np.transpose(np.where(image))
        hull = ConvexHull(points)
        deln = Delaunay(points[hull.vertices])
        idx = np.stack(np.indices(image.shape).astype('int16'), axis=-1)
        out_idx = np.nonzero(deln.find_simplex(idx) + 1)
        out_img = np.zeros(image.shape)
        out_img[out_idx] = 1
        return out_img, hull


class MergedROI:
    @classmethod
    def merged_roi(cls, ct_scan_path: str, anchor_segmentation_path: str, anchor_coordintes: tuple[int]):
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
        dss = DifficultBodySegmentation(orientation)
        body_segmentation = dss.isolate_body(data)
        body_segmentation_path = os.path.join(output_dir, f"{scan_basename}_body_segmentation.nii.gz")
        NiftyHandler.write(body_segmentation, body_segmentation_path, orientation)

        lungs_segmentation, bb, cc = dss.isolate_bs(body_segmentation)
        lungs_segmentation_path = os.path.join(output_dir, f"{scan_basename}_lungs_segmentation.nii.gz")
        NiftyHandler.write(lungs_segmentation, lungs_segmentation_path, orientation)
        body_lung_band = dss.three_d_band(body_segmentation, lungs_segmentation, bb, cc)

        spine_segmentation = SpineSegmentation(ct_scan_path, anchor_segmentation_path,
                                               anchor_coordintes).getSpineSegmentation()
        merged_segmentation = cls.__merge_band_and_spine(body_lung_band, spine_segmentation)
        merged_segmentation_path = os.path.join(output_dir, f"{scan_basename}_merged_segmentation.nii.gz")
        NiftyHandler.write(merged_segmentation, merged_segmentation_path, orientation)
        return merged_segmentation

    @staticmethod
    def __merge_band_and_spine(band_3d: np.ndarray, spine_segmentation: np.ndarray) -> np.ndarray:
        """
        merge the 3d band together with the spine segmentation and draw a bounding box around the spine.
        Parameters
        ----------
        band_3d
        spine_segmentation

        Returns
        -------

        """
        band_slices = np.where(np.any(band_3d, axis=(0, 1)))
        sliced_spine = np.zeros(spine_segmentation.shape)
        sliced_spine[:, :, band_slices] = spine_segmentation[:, :, band_slices]
        boxed_spine = MergedROI.draw_bounding_box_3d(sliced_spine)
        band_and_spine = ((band_3d > 0).astype(int) | (boxed_spine > 0).astype(int)).astype(float)
        return band_and_spine

    @staticmethod
    def draw_bounding_box_3d(binary_image: np.ndarray) -> np.ndarray:
        """
        Draws a bounding box around the single connected component in a 3D binary image.

        Parameters:
            binary_image (ndarray): A 3D binary image containing a single connected component.

        Returns:
            ndarray: A copy of the input image with a bounding box drawn around the connected component.
        """

        # Get the coordinates of all non-zero voxels in the image
        non_zero_voxels = np.nonzero(binary_image)

        # Find the minimum and maximum coordinates in each dimension
        min_x, max_x = np.min(non_zero_voxels[0]), np.max(non_zero_voxels[0])
        min_y, max_y = np.min(non_zero_voxels[1]), np.max(non_zero_voxels[1])
        min_z, max_z = np.min(non_zero_voxels[2]), np.max(non_zero_voxels[2])

        # Create a copy of the input image with the bounding box drawn around the connected component
        output_image = np.zeros_like(binary_image)
        output_image[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1] = 1
        output_image += binary_image

        return output_image.astype(float)
