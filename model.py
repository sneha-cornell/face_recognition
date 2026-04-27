import tensorflow as tf
from tensorflow.keras import layers, Model


def expanded_dw_pw_block(x, out_filters, expansion=2, stride=1, name_prefix=""):
    """
    Expand → DepthWise → linear project block (no residual).

    Compared to plain DW+PW:
      - The expand 1x1 widens channels by `expansion` before the DW conv,
        giving it a richer space to detect spatial patterns in.
      - The project 1x1 compresses back to out_filters with NO activation
        (linear projection) to avoid zeroing out values when collapsing a
        high-dimensional space down to a narrow one.
      - No residual Add — safe for GPX10 until Add support is confirmed.

    When expansion=1 the expand step is skipped and this reduces to a
    plain DW+PW block (used for the first block where channels are small).
    """
    in_ch = x.shape[-1]
    expanded_ch = in_ch * expansion

    # Expand: widen channels so DW conv works in a richer space
    if expansion > 1:
        x = layers.Conv2D(
            expanded_ch, 1, use_bias=False, name=f"{name_prefix}_expand"
        )(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_expand_bn")(x)
        x = layers.ReLU(name=f"{name_prefix}_expand_relu")(x)

    # DepthWise: spatial feature extraction in the expanded space
    x = layers.DepthwiseConv2D(
        kernel_size=3, strides=stride, padding="same", use_bias=False,
        name=f"{name_prefix}_dw",
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_dw_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_dw_relu")(x)

    # Linear project: compress back to out_filters, no activation
    x = layers.Conv2D(
        out_filters, 1, use_bias=False, name=f"{name_prefix}_project"
    )(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_project_bn")(x)
    # No ReLU here — linear projection preserves the compressed representation

    return x


def build_face_backbone(embedding_dim=128, input_shape=(112, 112, 3)):
    """
    Lightweight face recognition backbone — expanded DW+PW variant.

    Middle ground between plain DW+PW and full MobileFaceNet:
      - Expand+DW+linear-project blocks (no residual Add required)
      - expansion=1 on block1 (channels too small to benefit)
      - expansion=2 on all other blocks (DW sees 2x channels)

    GPX10 compiler constraints respected:
      - DepthWise-only groups
      - Dilation = 1
      - ReLU activations only (fused into Conv)
      - BatchNorm supported
      - AveragePool2D supported
      - No element-wise Add

    Approximate param count: ~404K

    Spatial progression:
      112x112 -> 56x56 -> 28x28 -> 14x14 -> 7x7 -> 1x1 -> embedding
    """
    inputs = tf.keras.Input(shape=input_shape, name="input_face")

    # 112x112x3 -> 56x56x32
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # expansion=1: channels too small (32) for expansion to help meaningfully
    # 56x56x32 -> 56x56x32
    x = expanded_dw_pw_block(x, 32,  expansion=1, stride=1, name_prefix="block1")
    # 56x56x32 -> 28x28x64
    x = expanded_dw_pw_block(x, 64,  expansion=2, stride=2, name_prefix="block2")
    # 28x28x64 -> 28x28x64
    x = expanded_dw_pw_block(x, 64,  expansion=2, stride=1, name_prefix="block3")
    # 28x28x64 -> 28x28x128
    x = expanded_dw_pw_block(x, 128, expansion=2, stride=1, name_prefix="block4")
    # 28x28x128 -> 14x14x128
    x = expanded_dw_pw_block(x, 128, expansion=2, stride=2, name_prefix="block5")
    # 14x14x128 -> 14x14x128
    x = expanded_dw_pw_block(x, 128, expansion=2, stride=1, name_prefix="block6")
    # 14x14x128 -> 14x14x128
    x = expanded_dw_pw_block(x, 128, expansion=2, stride=1, name_prefix="block7")
    # 14x14x128 -> 7x7x256  (project to 256 here — no separate conv_proj needed)
    x = expanded_dw_pw_block(x, 256, expansion=2, stride=2, name_prefix="block8")

    # Global average pool: 7x7x256 -> 1x1x256
    x = layers.AveragePooling2D(pool_size=7, strides=1, padding="valid", name="gap")(x)
    x = layers.Flatten(name="flatten")(x)

    # Linear embedding head + BN — BN keeps norms bounded so ArcFace stays well-conditioned
    embeddings = layers.Dense(embedding_dim, use_bias=False, name="embedding")(x)
    embeddings = layers.BatchNormalization(name="embedding_bn")(embeddings)

    return Model(inputs=inputs, outputs=embeddings, name="face_backbone")
