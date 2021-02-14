import tensorflow as tf
import numpy as np


class WSConv2D(tf.keras.layers.Conv2D):
    """WSConv2d
    Reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """

    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(
            kernel_initializer="he_normal", *args, **kwargs
        )

    def standardize_weight(self, weight, eps):
        mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)
        var = tf.math.reduce_variance(weight, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        # Get gain
        gain = self.add_weight(
            name='gain',
            shape=(weight.shape[-1],),
            initializer="ones",
            trainable=True,
            dtype=self.dtype)
        # Manually fused normalization, eq. to (w - mean) * gain / sqrt(N * var)
        scale = tf.math.rsqrt(
            tf.math.maximum(
                var * fan_in,
                tf.convert_to_tensor(eps, dtype=self.dtype)
            )
        ) * gain
        shift = mean * scale
        return weight * scale - shift

    def call(self, inputs, eps=1e-4):
        self.kernel.assign(self.standardize_weight(self.kernel, eps))
        return super().call(inputs)