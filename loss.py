import math
import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
    """
    ArcFace: Additive Angular Margin Loss.

    Adds an angular margin m to the target class angle before computing
    softmax, pushing intra-class embeddings closer and inter-class embeddings
    further apart in hyperspherical space.

    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for
    Deep Face Recognition", CVPR 2019.

    Usage:
      - During training: pass both embeddings and labels.
      - During inference: only the backbone is used; this layer is discarded.

    Args:
      num_classes: Number of identities in the training dataset.
      margin: Angular margin in radians. Default 0.5 (~28.6 degrees).
      scale: Feature scale (s). Default 64.0.
    """

    def __init__(self, num_classes, margin=0.5, scale=64.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Precompute margin trig values
        self.cos_m = tf.constant(math.cos(margin), dtype=tf.float32)
        self.sin_m = tf.constant(math.sin(margin), dtype=tf.float32)
        # Stability threshold: if cos(theta) < cos(pi - m), use linear approximation
        self.threshold = tf.constant(math.cos(math.pi - margin), dtype=tf.float32)
        self.mm = tf.constant(math.sin(math.pi - margin) * margin, dtype=tf.float32)

    def build(self, input_shape):
        embedding_dim = input_shape[-1]
        self.W = self.add_weight(
            name="arcface_W",
            shape=(embedding_dim, self.num_classes),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, embeddings, labels=None, training=False):
        # Normalize embeddings and class weights onto unit hypersphere
        emb_norm = tf.nn.l2_normalize(embeddings, axis=1)       # (B, d)
        W_norm = tf.nn.l2_normalize(self.W, axis=0)             # (d, C)

        # Cosine similarity between each embedding and each class center
        cosine = tf.matmul(emb_norm, W_norm)                    # (B, C)

        if not training or labels is None:
            return cosine * self.scale

        # sin(theta) from cos(theta): sqrt(1 - cos^2), clamped for stability
        sine = tf.sqrt(tf.maximum(1.0 - tf.square(cosine), 1e-7))

        # cos(theta + m) = cos*cos_m - sin*sin_m
        cos_theta_m = cosine * self.cos_m - sine * self.sin_m

        # If theta + m > pi (cos_theta < threshold), fall back to cos(theta) - mm
        # to avoid producing logits that push the wrong direction
        cos_theta_m = tf.where(cosine > self.threshold, cos_theta_m, cosine - self.mm)

        # Apply margin only to the ground-truth class
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = tf.where(tf.cast(one_hot, tf.bool), cos_theta_m, cosine)

        return logits * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "margin": self.margin,
            "scale": self.scale,
        })
        return config
