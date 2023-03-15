import os
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
import numpy as np


class NiftyHandler:
    """
    Class responsible for the reading and writing of nifti files.
    """

    @classmethod
    def read(cls, nifti_file_path: str):
        """
        reads the nifti file that is in a gz format.
        :param nifti_file_path:
        :return: data and orientation data
        """

        nifti_file_path = cls.correct_filename(nifti_file_path)
        if not os.path.exists(nifti_file_path):
            raise ValueError(f"File name {nifti_file_path} does not exist")
        nifti_file = nib.load(nifti_file_path)  # might raise an exception

        original_orientation = io_orientation(nifti_file.affine)
        ras_ornt = axcodes2ornt("RAS")

        to_canonical = original_orientation  # Same as ornt_transform(img_ornt, ras_ornt)
        from_canonical = ornt_transform(ras_ornt, original_orientation)

        # Same as as_closest_canonical
        img_canonical = nifti_file.as_reoriented(to_canonical)
        img_data = img_canonical.get_fdata()
        return img_data, from_canonical

    @classmethod
    def write(cls, data, file_name: str, orientation=None):
        """
        :param data: ndarray of the data
        :param nifti_file: nifti file that was opened
        :param file_name: file name for saving
        :return:
        """

        file_name = cls.correct_filename(file_name)
        new_nifti = nib.Nifti1Image(data, np.eye(4))
        new_nifti = new_nifti.as_reoriented(orientation)
        nib.save(new_nifti, file_name)

    @staticmethod
    def correct_filename(file_name):
        extension = ".nii.gz"
        if os.path.basename(file_name).endswith(extension):
            return file_name
        else:
            return file_name + extension
