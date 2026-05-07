"""
cnn_scratch.py
--------------
Stage 4 — CNN from Scratch for FER2013 (7-class facial expression recognition)

Architecture: 4 Convolutional Blocks + Fully Connected Head
Input: (48, 48, 1) grayscale images, pixel values in [0, 1]

Block Layout:
  Block 1: Conv(32)  -> BN -> ReLU -> Conv(32)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  Block 2: Conv(64)  -> BN -> ReLU -> Conv(64)  -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  Block 3: Conv(128) -> BN -> ReLU -> Conv(128) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  Block 4: Conv(256) -> BN -> ReLU -> Conv(256) -> BN -> ReLU -> MaxPool -> Dropout(0.25)
  Head   : Flatten -> Dense(512) -> BN -> ReLU -> Dropout(0.5)
                   -> Dense(256) -> ReLU -> Dropout(0.3)
                   -> Dense(7, softmax)

Spatial dimensions (48x48 input):
  After Block 1 MaxPool: 24 x 24 x 32
  After Block 2 MaxPool: 12 x 12 x 64
  After Block 3 MaxPool:  6 x  6 x 128
  After Block 4 MaxPool:  3 x  3 x 256
  Flatten: 2,304 units

Total trainable parameters: ~3.4M
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense,
)


NUM_CLASSES  = 7
INPUT_SHAPE  = (48, 48, 1)


def conv_bn_relu(x, filters: int, kernel_size: int = 3):
    """Convolution -> Batch Normalization -> ReLU activation block."""
    x = Conv2D(
        filters     = filters,
        kernel_size = kernel_size,
        padding     = "same",
        use_bias    = False,          # BN handles bias
        kernel_initializer = "he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def build_cnn_scratch(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int   = NUM_CLASSES,
    dropout_conv: float = 0.25,
    dropout_fc1:  float = 0.50,
    dropout_fc2:  float = 0.30,
) -> Model:
    """
    Build and return the CNN-from-scratch Keras functional model.

    Args:
        input_shape  : Image shape, default (48, 48, 1).
        num_classes  : Number of output classes, default 7.
        dropout_conv : Dropout rate after each conv block.
        dropout_fc1  : Dropout rate after first dense layer.
        dropout_fc2  : Dropout rate after second dense layer.

    Returns:
        Uncompiled Keras Model.
    """
    inputs = Input(shape=input_shape, name="input_image")

    # ── Convolutional Block 1 — 32 filters ───────────────────────────────────
    x = conv_bn_relu(inputs, 32)
    x = conv_bn_relu(x,      32)
    x = MaxPooling2D(pool_size=(2, 2), name="maxpool_1")(x)
    x = Dropout(dropout_conv, name="drop_conv_1")(x)
    # Output: (24, 24, 32)

    # ── Convolutional Block 2 — 64 filters ───────────────────────────────────
    x = conv_bn_relu(x, 64)
    x = conv_bn_relu(x, 64)
    x = MaxPooling2D(pool_size=(2, 2), name="maxpool_2")(x)
    x = Dropout(dropout_conv, name="drop_conv_2")(x)
    # Output: (12, 12, 64)

    # ── Convolutional Block 3 — 128 filters ──────────────────────────────────
    x = conv_bn_relu(x, 128)
    x = conv_bn_relu(x, 128)
    x = MaxPooling2D(pool_size=(2, 2), name="maxpool_3")(x)
    x = Dropout(dropout_conv, name="drop_conv_3")(x)
    # Output: (6, 6, 128)

    # ── Convolutional Block 4 — 256 filters ──────────────────────────────────
    x = conv_bn_relu(x, 256)
    x = conv_bn_relu(x, 256)
    x = MaxPooling2D(pool_size=(2, 2), name="maxpool_4")(x)
    x = Dropout(dropout_conv, name="drop_conv_4")(x)
    # Output: (3, 3, 256) -> Flatten = 2,304 units

    # ── Fully Connected Head ──────────────────────────────────────────────────
    x = Flatten(name="flatten")(x)

    x = Dense(512, use_bias=False, kernel_initializer="he_normal", name="fc_1")(x)
    x = BatchNormalization(name="bn_fc_1")(x)
    x = Activation("relu", name="relu_fc_1")(x)
    x = Dropout(dropout_fc1, name="drop_fc_1")(x)

    x = Dense(256, activation="relu", kernel_initializer="he_normal", name="fc_2")(x)
    x = Dropout(dropout_fc2, name="drop_fc_2")(x)

    # ── Output Layer ──────────────────────────────────────────────────────────
    outputs = Dense(num_classes, activation="softmax", name="output_softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FER_CNN_Scratch")
    return model


def compile_model(model: Model, learning_rate: float = 1e-3) -> Model:
    """Compile with Adam optimizer and categorical cross-entropy."""
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy"],
    )
    return model


if __name__ == "__main__":
    model = build_cnn_scratch()
    model = compile_model(model)
    model.summary()
    print(f"\nTotal params        : {model.count_params():,}")
    print(f"Trainable params    : {sum(w.numpy().size for w in model.trainable_weights):,}")
