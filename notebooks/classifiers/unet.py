import torch
import cv2
import numpy as np
from notebooks.unet.unet import UNet


def mask_parse(mask):
    """Function used to change mask from (512, 512, 1) to (512, 512, 3)

    Parameters
    ----------
    mask: np.ndarray
        mask to be changed

    Returns
    -------
    np.ndarray
        changed mask
    """
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


class UNetClassifier:
    """Class used to make predictions based on uploaded `UNet` model"""

    def __init__(self, model_path):
        """
        Parameters
        ----------
        model_path: str
            Path to the file with `.pth` extension with model
        """
        self.model = UNet(3)
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.model.eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Function used to predict the image

        Parameters
        ----------
        image: np.ndarray
            Matrix representing the image to be made predictions upon

        Returns
        -------
        np.ndarray
            Predicted map with detected blood vessels
        """
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        img = image / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img.astype(np.float32))

        with torch.no_grad():
            pred_y = self.model(img)

            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        return pred_y
