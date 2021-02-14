import tensorflow as tf
import numpy as np

from base import WSConv2D, SqueezeExcite, StochDepth


class NFBlock(tf.keras.Model):
    """Normalizer-Free Net Block."""

    def __init__(self, in_ch, out_ch, expansion=0.5, se_ratio=0.5,
        kernel_shape=3, group_size=128, stride=1,
        beta=1.0, alpha=0.2,
        which_conv=WSConv2D, activation=tf.keras.activations.gelu,
        big_width=True, use_two_convs=True,
        stochdepth_rate=None, name=None
    ):
        super(NFBlock, self).__init__(name=name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.kernel_shape = kernel_shape
        self.activation = activation
        self.beta, self.alpha = beta, alpha
        # Mimic resnet style bigwidth scaling?
        width = int((self.out_ch if big_width else self.in_ch) * expansion)
        # Round expanded with based on group count
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride
        self.use_two_convs = use_two_convs
        # Conv 0 (typically expansion conv)
        self.conv0 = which_conv(
            filters=self.width, kernel_size=1, padding='same', name='conv0'
        )
        # Grouped NxN conv
        self.conv1 = which_conv(
            filters=self.width, kernel_size=kernel_shape, strides=stride,
            padding='same', groups=self.groups, name='conv1')
        if self.use_two_convs:
            self.conv1b = which_conv(
                filters=self.width, kernel_size=kernel_shape, strides=1, padding='same',
                groups=self.groups, name='conv1b'
            )
        # Conv 2, typically projection conv
        self.conv2 = which_conv(
            filters=self.out_ch, kernel_size=1, padding='same', name='conv2'
        )
        # Use shortcut conv on channel change or downsample.
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.conv_shortcut = which_conv(
                filters=self.out_ch, kernel_size=1, padding='same', name='conv_shortcut'
            )
        # Squeeze + Excite Module
        self.se = SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)
        # Are we using stochastic depth?
        self._has_stochdepth = (
            stochdepth_rate is not None
            and stochdepth_rate > 0.0
            and stochdepth_rate < 1.0
        )
        if self._has_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)

    def call(self, x, training):
        out = self.activation(x) * self.beta
        if self.stride > 1:  # Average-pool downsample.
            shortcut = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding='same'
            )(out)
            if self.use_projection:
                shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x
        out = self.conv0(out)
        out = self.conv1(self.activation(out))
        if self.use_two_convs:
            out = self.conv1b(self.activation(out))
        out = self.conv2(self.activation(out))
        out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
        # Get average residual standard deviation for reporting metrics.
        res_avg_var = tf.math.reduce_mean(tf.math.reduce_variance(out, axis=[0, 1, 2]))
        # Apply stochdepth if applicable.
        if self._has_stochdepth:
            out = self.stoch_depth(out, training)
        # SkipInit Gain
        out = out * self.add_weight(
            name='skip_gain',
            shape=(),
            initializer="zeros",
            trainable=True,
            dtype=out.dtype
        )
        return out * self.alpha + shortcut, res_avg_var