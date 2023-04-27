import cv2
from typing import List, Tuple
import numpy


class ClassicClassifier:

    def __init__(self, kernel_sizes: List[Tuple[int, int]], contour_cutoff: int):
        """
        Parameters
        ----------
        kernel_sizes: List[Tuple[int, int]]
            Sizes of the kernels that will be used in morphology filter step
        contour_cutoff: int
            Contour areas bigger than this threshold will not be drawn on the mask
        """
        self.kernel_size = kernel_sizes
        self.contour_cutoff = contour_cutoff

    def process(self, image: numpy.ndarray) -> numpy.ndarray:
        """Function used to process an image.

        Processing consists of 8 steps.
        1. Separation data into RGB channels and extraction of the green channel for processing.
        2. Application of a CLAHE filter.
        3. Application of erosion and dilation (morphological filters) for each kernel from `kernel_sizes`
        4. Application of CLAHE filter.
        5. Application of image thresholding.
        6. Determination of the mask.
        7. Determination of the contours.
        8. Re-application of the thresholding.

        Useful links and materials used to write this function:
        1. CLAHE
            - https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html#gsc.tab=0
            - https://iq.opengenus.org/contrast-enhancement-algorithms/
        2. Morphological filters
            - https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        3. Contours
            - https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
            - https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0
            - https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

        Parameters
        ----------
        image: numpy.ndarray
            Input image to be processed

        Returns
        -------
        processed_image: numpy.ndarray
        """

        b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_clahe = clahe.apply(g)
        temporary_green = g_clahe

        for kernel_size in self.kernel_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            temporary_green = cv2.morphologyEx(temporary_green, cv2.MORPH_OPEN, kernel)
            temporary_green = cv2.morphologyEx(temporary_green, cv2.MORPH_CLOSE, kernel)

        proc_image = cv2.subtract(temporary_green, g_clahe)
        proc_image = clahe.apply(proc_image)

        _, threshold_image = cv2.threshold(proc_image, 15, 255, cv2.THRESH_BINARY)
        mask = numpy.ones(proc_image.shape[:2]) * 255

        contours, _ = cv2.findContours(threshold_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= 200:
                cv2.drawContours(mask, [contour], -1, 0, 1)

        proc_image = numpy.bitwise_and(proc_image.astype(int), mask.astype(int)).astype(float)
        _, proc_image = cv2.threshold(proc_image, 15, 255, cv2.THRESH_BINARY)

        return proc_image

