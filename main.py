import uuid
import os
from typing import Tuple

import cv2
import numpy as np

from flask import Flask, render_template, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, BooleanField
from werkzeug.utils import secure_filename

from notebooks.classifiers.classic import ClassicClassifier
from notebooks.classifiers.unet import UNetClassifier
from notebooks.utils.confusion_matrix import ConfusionMatrix

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY='dev',
)


class InputForm(FlaskForm):
    input_image = FileField("Prześlij plik",
                            validators=[FileRequired(), FileAllowed(['jpg', 'png'])])
    input_map = FileField("Prześlij mapę ekspercką (opcjonalne)",
                          validators=[FileAllowed(['jpg', 'png'])])
    submit = SubmitField()


def predict_classic(image_path: str, user_path: str, true_path: str = '') -> Tuple[str, ConfusionMatrix]:
    cc = ClassicClassifier([(5, 5), (12, 12), (20, 20)], 200)
    image = cv2.imread(image_path)
    pred_image = cc.predict(image)

    back = '\\'
    pred_path = f"{image_path.split(back)[-1].split('.')[0]}_pred_classic.png"
    pred_path = os.path.join(user_path, pred_path)
    cv2.imwrite(pred_path, pred_image)

    if true_path:
        true_image = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        cm = ConfusionMatrix(true_image.flatten(), pred_image.flatten())
        return pred_path, cm
    else:
        return pred_path, None


def predict_unet(image_path: str, user_path: str, true_path: str = '') -> Tuple[str, ConfusionMatrix]:

    def mask_parse(mask):
        mask = np.expand_dims(mask, axis=-1)  # (512, 512, 1)
        mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
        return mask

    uc = UNetClassifier('models/unet_checkpoint.pth')
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    y_pred = uc.predict(image)

    pred_image = mask_parse(y_pred)
    back = '\\'
    pred_path = f"{image_path.split(back)[-1].split('.')[0]}_pred_unet.png"
    pred_path = os.path.join(user_path, pred_path)
    cv2.imwrite(pred_path, pred_image * 255)

    if true_path:
        true_image = cv2.imread(true_path, cv2.IMREAD_GRAYSCALE)
        answer = cv2.resize(true_image, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        ans = answer / 255.0
        ans = ans[np.newaxis, ...]
        y_true = ans > 0.5
        y_true = y_true.astype(np.uint8)
        y_true = y_true.reshape(-1)

        cm = ConfusionMatrix(y_true, y_pred)

        return pred_path, cm
    else:
        return pred_path, None


@app.route("/", methods=['GET', 'POST'])
def home():
    form = InputForm()
    if form.validate_on_submit():
        # ------------------------------------------------------------------------------------------------------------ #
        # HANDLE UPLOADED FILES

        # get user's unique id
        uid = str(uuid.uuid4())
        user_path = os.path.join('static/temp', uid)
        os.makedirs(user_path)

        # get files from form
        f = form.input_image.data
        f_filename = secure_filename(f.filename)
        image_path = os.path.join(user_path, f_filename)
        f.save(image_path)

        f_m = form.input_map.data
        if f_m is not None:
            f_m_filename = secure_filename(f_m.filename)
            map_path = os.path.join(user_path, f_m_filename)
            f_m.save(map_path)

            classic_path, cm_classic = predict_classic(image_path, user_path, map_path)
            unet_path, cm_unet = predict_unet(image_path, user_path, map_path)
        else:
            classic_path, cm_classic = predict_classic(image_path, user_path)
            unet_path, cm_unet = predict_unet(image_path, user_path)

        # ------------------------------------------------------------------------------------------------------------ #
        # RESIZE UPLOADED FILES TO 512x512

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(image_path, img)

        if f_m is not None:
            ans = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
            ans = cv2.resize(ans, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(map_path, ans)

        pred = cv2.imread(classic_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.resize(pred, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(classic_path, pred)

        if f_m is not None:
            return render_template('result.html',
                                   image_path=image_path, map_path=map_path,
                                   classic_path=classic_path, unet_path=unet_path,
                                   cm_classic=cm_classic, cm_unet=cm_unet)
        else:
            return render_template('result.html',
                                   image_path=image_path,
                                   classic_path=classic_path, unet_path=unet_path,
                                   cm_classic=cm_classic, cm_unet=cm_unet)

    print(form.errors)
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')
