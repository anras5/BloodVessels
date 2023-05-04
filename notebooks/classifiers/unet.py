import torch
import cv2
import numpy as np
from notebooks.unet.unet import UNet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask


class UNetClassifier:

    def __init__(self, model_path):
        self.model = UNet(3)
        self.model.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.model.eval()

    def predict(self, image: np.ndarray) -> np.ndarray:
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
