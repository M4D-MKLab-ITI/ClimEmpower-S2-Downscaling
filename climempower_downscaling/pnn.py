"""Zero-shot Pansharpening Neural Network (PNN).

Reference: Masi G, Cozzolino D, Verdoliva L, et al. "Pansharpening by
convolutional neural networks." Remote Sensing, 2016, 8(7): 594.
Code reference: https://github.com/sergiovitale/pansharpening-cnn-python-version

This is a zero-shot method: for each scene, the real pan/multispectral pair is
degraded (via ``downgrade_images``) to synthesize a lower-resolution training
pair, a small CNN is trained on patches of that pair, and the trained network
is then applied at full resolution. As an alternative, ``run_pnn`` can load
pretrained weights and skip training entirely, provided the weights match the
model architecture built by ``build_pnn_model``.
"""

from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import convolve2d
import cv2
from tqdm import tqdm

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import Concatenate, Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .utils import upsample_interp23

_UINT16_MAX = 2**16 - 1


def downgrade_images(
    ms: np.ndarray, pan: np.ndarray, ratio: int
) -> tuple[np.ndarray, np.ndarray]:
    """Degrade a multispectral/pan pair by ``ratio`` with a Gaussian blur + decimate.

    Used to synthesize a lower-resolution training pair from a real scene, so the
    original full-resolution pair can serve as the training target.

    Parameters
    ----------
    ms : np.ndarray
        Multispectral stack, shape (m, n, C).
    pan : np.ndarray
        Panchromatic band, shape (M, N, 1) or (M, N).
    ratio : int
        Degradation factor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(ms_low_res, pan_low_res)``, shapes (m/ratio, n/ratio, C) and
        (M/ratio, N/ratio, 1).
    """
    ms = np.transpose(np.double(ms), (2, 0, 1))
    pan = np.squeeze(np.double(pan))
    ratio = np.double(ratio)

    sigma = (1 / (2 * 2.772587 / ratio**2)) ** 0.5
    kernel_1d = cv2.getGaussianKernel(9, sigma)
    kernel = kernel_1d @ kernel_1d.T

    ms_low_res_bands = []
    for band_idx in range(ms.shape[0]):
        blurred = convolve2d(ms[band_idx, :, :], kernel, mode="same", boundary="wrap")
        decimated = blurred[0 :: int(ratio), 0 :: int(ratio)]
        ms_low_res_bands.append(np.expand_dims(decimated, 0))
    ms_low_res = np.concatenate(ms_low_res_bands, axis=0)

    pan_blurred = convolve2d(pan, kernel, mode="same", boundary="wrap")
    pan_low_res = pan_blurred[0 :: int(ratio), 0 :: int(ratio)]

    ms_low_res = np.transpose(ms_low_res, (1, 2, 0))
    pan_low_res = np.expand_dims(pan_low_res, -1)
    return ms_low_res, pan_low_res


def psnr(y_true, y_pred):
    """Peak signal-to-noise ratio metric for uint16-scaled [0, 1] tensors."""
    mse = K.mean(K.square(y_true * _UINT16_MAX - y_pred * _UINT16_MAX), axis=(-3, -2, -1))
    return K.mean(20 * K.log(_UINT16_MAX / K.sqrt(mse)) / K.log(10.0))


def build_pnn_model(lrhs_size: tuple[int, int, int], hrms_size: tuple[int, int, int]) -> Model:
    """Build the 3-layer PNN architecture (Masi et al., 2016).

    Parameters
    ----------
    lrhs_size : tuple[int, int, int]
        Shape of the (upsampled) low-resolution multispectral patch, (H, W, C).
    hrms_size : tuple[int, int, int]
        Shape of the high-resolution panchromatic patch, (H, W, 1).

    Returns
    -------
    keras.Model
        Compiled two-input, single-output model.
    """
    lrhs_inputs = Input(lrhs_size)
    hrms_inputs = Input(hrms_size)

    mixed = Concatenate()([lrhs_inputs, hrms_inputs])
    mixed = Conv2D(64, (9, 9), strides=(1, 1), padding="same", activation="relu")(mixed)
    mixed = Conv2D(32, (5, 5), strides=(1, 1), padding="same", activation="relu")(mixed)
    output = Conv2D(
        lrhs_size[2], (5, 5), strides=(1, 1), padding="same", activation="relu"
    )(mixed)

    model = Model(inputs=[lrhs_inputs, hrms_inputs], outputs=output)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="mse", metrics=[psnr])
    return model


def train_pnn(
    hrms: np.ndarray,
    lrhs: np.ndarray,
    *,
    checkpoint_path: str | Path,
    epochs: int = 50,
    batch_size: int = 32,
    training_size: int = 32,
    stride: int = 8,
    verbose: int = 1,
) -> Model:
    """Train a PNN model zero-shot on patches synthesized from ``hrms``/``lrhs``.

    Parameters
    ----------
    hrms : np.ndarray
        High-resolution panchromatic band, shape (M, N, 1), normalized to [0, 1].
    lrhs : np.ndarray
        Low-resolution multispectral stack, shape (m, n, C), normalized to [0, 1].
    checkpoint_path : str | Path
        Where to save the best checkpoint (monitored on validation PSNR).
    epochs, batch_size, training_size, stride : int
        Training hyperparameters; ``training_size`` is the patch size and
        ``stride`` the patch sampling stride.
    verbose : int
        Keras verbosity level.

    Returns
    -------
    keras.Model
        The model with the training-size input shape (weights match the best
        checkpoint on disk).
    """
    m_rows, n_cols, c_bands = lrhs.shape
    hrms_rows, hrms_cols, hrms_bands = hrms.shape
    ratio = int(np.round(hrms_rows / m_rows))
    if ratio != int(np.round(hrms_cols / n_cols)):
        raise ValueError("hrms and lrhs must share the same row/column sharpening ratio")

    low_res_lrhs, low_res_hrms = downgrade_images(lrhs, hrms, ratio)
    low_res_lrhs = upsample_interp23(low_res_lrhs, ratio)

    hrhs_patches, hrms_patches, lrhs_patches = [], [], []
    for row in range(0, low_res_hrms.shape[0] - training_size, stride):
        for col in range(0, low_res_hrms.shape[1] - training_size, stride):
            hrhs_patches.append(lrhs[row : row + training_size, col : col + training_size, :])
            hrms_patches.append(
                low_res_hrms[row : row + training_size, col : col + training_size, :]
            )
            lrhs_patches.append(
                low_res_lrhs[row : row + training_size, col : col + training_size, :]
            )

    hrhs_patches = np.array(hrhs_patches, dtype="float16")
    hrms_patches = np.array(hrms_patches, dtype="float16")
    lrhs_patches = np.array(lrhs_patches, dtype="float16")

    shuffled_index = list(range(hrhs_patches.shape[0]))
    random.shuffle(shuffled_index)
    hrhs_patches = hrhs_patches[shuffled_index]
    hrms_patches = hrms_patches[shuffled_index]
    lrhs_patches = lrhs_patches[shuffled_index]

    def lr_schedule(epoch: int) -> float:
        lr = 5e-4
        if epoch > 40:
            lr *= 1e-2
        elif epoch > 20:
            lr *= 1e-1
        return lr

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        LearningRateScheduler(lr_schedule, verbose=verbose),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_psnr",
            mode="max",
            verbose=verbose,
            save_best_only=True,
        ),
    ]

    model = build_pnn_model(
        lrhs_size=(training_size, training_size, c_bands),
        hrms_size=(training_size, training_size, hrms_bands),
    )
    model.fit(
        x=[lrhs_patches, hrms_patches],
        y=hrhs_patches,
        validation_split=0.33,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
    )
    model.load_weights(str(checkpoint_path))
    return model


def reconstruct_pnn(
    model: Model,
    hrms: np.ndarray,
    lrhs: np.ndarray,
    ratio: int,
    *,
    testing_size: int = 400,
    reconstructing_size: int = 320,
) -> np.ndarray:
    """Run a trained PNN model over a full scene via overlap-tiled inference.

    Parameters
    ----------
    model : keras.Model
        A model built by ``build_pnn_model`` with input shapes matching
        ``(testing_size, testing_size, C)`` / ``(testing_size, testing_size, c)``.
    hrms : np.ndarray
        High-resolution panchromatic band, shape (M, N, 1), normalized to [0, 1].
    lrhs : np.ndarray
        Low-resolution multispectral stack, shape (m, n, C), normalized to [0, 1].
    ratio : int
        Sharpening ratio between ``hrms`` and ``lrhs``.
    testing_size, reconstructing_size : int
        Tile size fed to the model and the centered region of each tile kept
        in the output, respectively. The gap between them is padding used to
        avoid border artifacts at tile edges.

    Returns
    -------
    np.ndarray
        Fused image, dtype uint16, shape (min(M, m*ratio), min(N, n*ratio), C).
    """
    hrms_rows, hrms_cols, _ = hrms.shape
    lrhs_rows, lrhs_cols, c_bands = lrhs.shape
    left_pad = (testing_size - reconstructing_size) // 2

    out_rows = min(hrms_rows, lrhs_rows * ratio)
    out_cols = min(hrms_cols, lrhs_cols * ratio)

    output = np.zeros((out_rows, out_cols, c_bands), dtype="uint16")

    used_lrhs = lrhs[: out_rows // ratio, : out_cols // ratio, :]
    used_hrms = hrms[:out_rows, :out_cols, :]
    used_lrhs = upsample_interp23(used_lrhs, ratio)

    used_lrhs = np.expand_dims(used_lrhs, 0)
    used_hrms = np.expand_dims(used_hrms, 0)
    used_lrhs = np.pad(
        used_lrhs,
        ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
        mode="symmetric",
    )
    used_hrms = np.pad(
        used_hrms,
        ((0, 0), (left_pad, testing_size), (left_pad, testing_size), (0, 0)),
        mode="symmetric",
    )

    for row in tqdm(range(0, out_rows, reconstructing_size)):
        for col in range(0, out_cols, reconstructing_size):
            tile_lrhs = used_lrhs[:, row : row + testing_size, col : col + testing_size, :]
            tile_hrms = used_hrms[:, row : row + testing_size, col : col + testing_size, :]

            prediction = model.predict([tile_lrhs, tile_hrms], verbose=0)
            prediction = np.clip(prediction, 0, 1)
            prediction.shape = (testing_size, testing_size, c_bands)
            prediction = prediction[
                left_pad : testing_size - left_pad, left_pad : testing_size - left_pad
            ]
            prediction = np.uint16(prediction * _UINT16_MAX)

            if row + reconstructing_size > out_rows:
                prediction = prediction[: out_rows - row, :, :]
            if col + reconstructing_size > out_cols:
                prediction = prediction[:, : out_cols - col, :]

            output[row : row + reconstructing_size, col : col + reconstructing_size] = prediction

    return output


def run_pnn(
    hrms: np.ndarray,
    lrhs: np.ndarray,
    *,
    weights_path: Optional[str | Path] = None,
    checkpoint_path: str | Path = "pnn_checkpoint.keras",
    epochs: int = 50,
    batch_size: int = 32,
    testing_size: int = 400,
    reconstructing_size: int = 320,
) -> np.ndarray:
    """Pansharpen ``lrhs`` with PNN, either training zero-shot or loading weights.

    Parameters
    ----------
    hrms : np.ndarray
        High-resolution panchromatic band, shape (M, N, 1), normalized to [0, 1].
    lrhs : np.ndarray
        Low-resolution multispectral stack, shape (m, n, C), normalized to [0, 1].
    weights_path : str | Path, optional
        If given, skip training and load these weights into a model sized for
        full-scene inference. The weights must have been trained with the same
        number of bands (``C``) and pan channels as ``lrhs``/``hrms``.
    checkpoint_path : str | Path
        Where to save the best checkpoint during training (ignored if
        ``weights_path`` is given).
    epochs, batch_size : int
        Training hyperparameters (ignored if ``weights_path`` is given).
    testing_size, reconstructing_size : int
        Overlap-tiling parameters for full-scene inference, see ``reconstruct_pnn``.

    Returns
    -------
    np.ndarray
        Fused image, dtype uint16.
    """
    m_rows, n_cols, c_bands = lrhs.shape
    hrms_rows, hrms_cols, hrms_bands = hrms.shape
    ratio = int(np.round(hrms_rows / m_rows))
    if ratio != int(np.round(hrms_cols / n_cols)):
        raise ValueError("hrms and lrhs must share the same row/column sharpening ratio")

    if weights_path is not None:
        model = build_pnn_model(
            lrhs_size=(testing_size, testing_size, c_bands),
            hrms_size=(testing_size, testing_size, hrms_bands),
        )
        model.load_weights(str(weights_path))
    else:
        model = train_pnn(
            hrms,
            lrhs,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            batch_size=batch_size,
        )
        model = build_pnn_model(
            lrhs_size=(testing_size, testing_size, c_bands),
            hrms_size=(testing_size, testing_size, hrms_bands),
        )
        model.load_weights(str(checkpoint_path))

    fused = reconstruct_pnn(
        model,
        hrms,
        lrhs,
        ratio,
        testing_size=testing_size,
        reconstructing_size=reconstructing_size,
    )

    K.clear_session()
    gc.collect()
    del model

    return fused
