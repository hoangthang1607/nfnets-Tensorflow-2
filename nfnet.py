import tensorflow as tf

from base import WSConv2D, SqueezeExcite, StochDepth


nfnet_params = {}


# F-series models
nfnet_params.update(
    **{
        "F0": {
            "width": [256, 512, 1536, 1536],
            "depth": [1, 2, 6, 3],
            "train_imsize": 192,
            "test_imsize": 256,
            "RA_level": "405",
            "drop_rate": 0.2,
        },
        "F1": {
            "width": [256, 512, 1536, 1536],
            "depth": [2, 4, 12, 6],
            "train_imsize": 224,
            "test_imsize": 320,
            "RA_level": "410",
            "drop_rate": 0.3,
        },
        "F2": {
            "width": [256, 512, 1536, 1536],
            "depth": [3, 6, 18, 9],
            "train_imsize": 256,
            "test_imsize": 352,
            "RA_level": "410",
            "drop_rate": 0.4,
        },
        "F3": {
            "width": [256, 512, 1536, 1536],
            "depth": [4, 8, 24, 12],
            "train_imsize": 320,
            "test_imsize": 416,
            "RA_level": "415",
            "drop_rate": 0.4,
        },
        "F4": {
            "width": [256, 512, 1536, 1536],
            "depth": [5, 10, 30, 15],
            "train_imsize": 384,
            "test_imsize": 512,
            "RA_level": "415",
            "drop_rate": 0.5,
        },
        "F5": {
            "width": [256, 512, 1536, 1536],
            "depth": [6, 12, 36, 18],
            "train_imsize": 416,
            "test_imsize": 544,
            "RA_level": "415",
            "drop_rate": 0.5,
        },
        "F6": {
            "width": [256, 512, 1536, 1536],
            "depth": [7, 14, 42, 21],
            "train_imsize": 448,
            "test_imsize": 576,
            "RA_level": "415",
            "drop_rate": 0.5,
        },
        "F7": {
            "width": [256, 512, 1536, 1536],
            "depth": [8, 16, 48, 24],
            "train_imsize": 480,
            "test_imsize": 608,
            "RA_level": "415",
            "drop_rate": 0.5,
        },
    }
)

# Minor variants FN+, slightly wider
nfnet_params.update(
    **{
        **{
            f"{key}+": {
                **nfnet_params[key],
                "width": [384, 768, 2048, 2048],
            }
            for key in nfnet_params
        }
    }
)


# Nonlinearities with magic constants (gamma) baked in.
# Note that not all nonlinearities will be stable, especially if they are
# not perfectly monotonic. Good choices include relu, silu, and gelu.
nonlinearities = {
    "identity": lambda x: x,
    "celu": lambda x: tf.nn.crelu(x) * 1.270926833152771,
    "elu": lambda x: tf.keras.activations.elu(x) * 1.2716004848480225,
    "gelu": lambda x: tf.keras.activations.gelu(x) * 1.7015043497085571,
    #     'glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    "leaky_relu": lambda x: tf.nn.leaky_relu(x) * 1.70590341091156,
    "log_sigmoid": lambda x: tf.math.log(tf.nn.sigmoid(x)) * 1.9193484783172607,
    "log_softmax": lambda x: tf.math.log(tf.nn.softmax(x)) * 1.0002083778381348,
    "relu": lambda x: tf.keras.activations.relu(x) * 1.7139588594436646,
    "relu6": lambda x: tf.nn.relu6(x) * 1.7131484746932983,
    "selu": lambda x: tf.keras.activations.selu(x) * 1.0008515119552612,
    "sigmoid": lambda x: tf.keras.activations.sigmoid(x) * 4.803835391998291,
    "silu": lambda x: tf.nn.silu(x) * 1.7881293296813965,
    "soft_sign": lambda x: tf.nn.softsign(x) * 2.338853120803833,
    "softplus": lambda x: tf.keras.activations.softplus(x) * 1.9203323125839233,
    "tanh": lambda x: tf.keras.activations.tanh(x) * 1.5939117670059204,
}


class NFNet(tf.keras.Model):
    """Normalizer-Free Networks with an improved architecture.
    References:
    [Brock, Smith, De, Simonyan 2021] High-Performance Large-Scale Image
    Recognition Without Normalization.
    """

    variant_dict = nfnet_params

    def __init__(
        self,
        num_classes,
        variant="F0",
        width=1.0,
        se_ratio=0.5,
        alpha=0.2,
        stochdepth_rate=0.1,
        drop_rate=None,
        activation="gelu",
        fc_init=None,
        final_conv_mult=2,
        final_conv_ch=None,
        use_two_convs=True,
        name="NFNet",
        label_smoothing=0.1,
        ema_decay=0.99999,
        clipping_factor=0.01,
    ):
        super(NFNet, self).__init__(name=name)
        self.num_classes = num_classes
        self.variant = variant
        self.width = width
        self.se_ratio = se_ratio
        # Get variant info
        block_params = self.variant_dict[self.variant]
        self.train_imsize = block_params["train_imsize"]
        self.test_imsize = block_params["test_imsize"]
        self.width_pattern = block_params["width"]
        self.depth_pattern = block_params["depth"]
        self.bneck_pattern = block_params.get("expansion", [0.5] * 4)
        self.group_pattern = block_params.get("group_width", [128] * 4)
        self.big_pattern = block_params.get("big_width", [True] * 4)
        self.activation = nonlinearities[activation]
        if drop_rate is None:
            self.drop_rate = block_params["drop_rate"]
        else:
            self.drop_rate = drop_rate
        self.which_conv = WSConv2D
        self.spositives = tf.convert_to_tensor(
            1.0 - label_smoothing, dtype=tf.float32
        )
        self.snegatives = tf.convert_to_tensor(
            label_smoothing / num_classes, dtype=tf.float32
        )
        self.ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
        self.clipping_factor = clipping_factor
        # Stem
        ch = self.width_pattern[0] // 2
        self.stem = tf.keras.Sequential(
            [
                self.which_conv(
                    16,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    name="stem_conv0",
                ),
                tf.keras.layers.Lambda(self.activation, name="act_stem_conv0"),
                self.which_conv(
                    32,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    name="stem_conv1",
                ),
                tf.keras.layers.Lambda(self.activation, name="act_stem_conv1"),
                self.which_conv(
                    64,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    name="stem_conv2",
                ),
                tf.keras.layers.Lambda(self.activation, name="act_stem_conv2"),
                self.which_conv(
                    ch,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    name="stem_conv3",
                ),
            ]
        )

        # Body
        self.blocks = []
        expected_std = 1.0
        num_blocks = sum(self.depth_pattern)
        index = 0  # Overall block index
        stride_pattern = [1, 2, 2, 2]
        block_args = zip(
            self.width_pattern,
            self.depth_pattern,
            self.bneck_pattern,
            self.group_pattern,
            self.big_pattern,
            stride_pattern,
        )
        for (
            block_width,
            stage_depth,
            expand_ratio,
            group_size,
            big_width,
            stride,
        ) in block_args:
            for block_index in range(stage_depth):
                # Scalar pre-multiplier so each block sees an N(0,1) input at init
                beta = 1.0 / expected_std
                # Block stochastic depth drop-rate
                block_stochdepth_rate = stochdepth_rate * index / num_blocks
                out_ch = int(block_width * self.width)
                self.blocks += [
                    NFBlock(
                        ch,
                        out_ch,
                        expansion=expand_ratio,
                        se_ratio=se_ratio,
                        group_size=group_size,
                        stride=stride if block_index == 0 else 1,
                        beta=beta,
                        alpha=alpha,
                        activation=self.activation,
                        which_conv=self.which_conv,
                        stochdepth_rate=block_stochdepth_rate,
                        big_width=big_width,
                        use_two_convs=use_two_convs,
                    )
                ]
                ch = out_ch
                index += 1
                # Reset expected std but still give it 1 block of growth
                if block_index == 0:
                    expected_std = 1.0
                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        # Head
        if final_conv_mult is None:
            if final_conv_ch is None:
                raise ValueError(
                    "Must provide one of final_conv_mult or final_conv_ch"
                )
            ch = final_conv_ch
        else:
            ch = int(final_conv_mult * ch)
        self.final_conv = self.which_conv(
            ch, kernel_size=1, padding="same", name="final_conv"
        )
        # By default, initialize with N(0, 0.01)
        if fc_init is None:
            fc_init = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)
        self.fc = tf.keras.layers.Dense(
            self.num_classes, kernel_initializer=fc_init, use_bias=True
        )

    def call(self, x, training=True):
        """Return the output of the final layer without any [log-]softmax."""
        # Stem
        outputs = {}
        out = self.stem(x)
        # Blocks
        for i, block in enumerate(self.blocks):
            out, res_avg_var = block(out, training=training)
        # Final-conv->activation, pool, dropout, classify
        out = tf.keras.layers.Lambda(self.activation)(self.final_conv(out))
        pool = tf.math.reduce_mean(out, [1, 2])
        outputs["pool"] = pool
        # Optionally apply dropout
        if self.drop_rate > 0.0 and training:
            pool = tf.keras.layers.Dropout(self.drop_rate)(pool)
        outputs["logits"] = self.fc(pool)
        return outputs

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_true = self.spositives * y_true + self.snegatives
            loss_values = self.compiled_loss(y_true, y_pred["logits"])

        gradients = tape.gradient(loss_values, self.trainable_weights)
        clipped_gradients = [
            grad
            if ("dense" in weight.name and "squeeze_excite" not in weight.name)
            else clip_gradient(grad, weight, clipping=self.clipping_factor)
            for grad, weight in zip(gradients, self.trainable_weights)
        ]
        # https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/SGDW
        opt_op = self.optimizer.apply_gradients(
            zip(clipped_gradients, self.trainable_weights),
            decay_var_list=[
                layer
                for layer in self.trainable_weights
                if not ("gain" in layer.name or "bias" in layer.name)
            ],
        )
        with tf.control_dependencies([opt_op]):
            self.ema.apply(self.trainable_variables)
        self.compiled_metrics.update_state(y_true, y_pred["logits"])

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        # Unpack the data
        x, y_true = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y_true, y_pred["logits"])
        # Update the metrics.
        self.compiled_metrics.update_state(y_true, y_pred["logits"])
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


class NFBlock(tf.keras.Model):
    """Normalizer-Free Net Block."""

    def __init__(
        self,
        in_ch,
        out_ch,
        expansion=0.5,
        se_ratio=0.5,
        kernel_shape=3,
        group_size=128,
        stride=1,
        beta=1.0,
        alpha=0.2,
        which_conv=WSConv2D,
        activation=tf.keras.activations.gelu,
        big_width=True,
        use_two_convs=True,
        stochdepth_rate=None,
        name=None,
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
            filters=self.width, kernel_size=1, padding="same", name="conv0"
        )
        # Grouped NxN conv
        self.conv1 = which_conv(
            filters=self.width,
            kernel_size=kernel_shape,
            strides=stride,
            padding="same",
            groups=self.groups,
            name="conv1",
        )
        if self.use_two_convs:
            self.conv1b = which_conv(
                filters=self.width,
                kernel_size=kernel_shape,
                strides=1,
                padding="same",
                groups=self.groups,
                name="conv1b",
            )
        # Conv 2, typically projection conv
        self.conv2 = which_conv(
            filters=self.out_ch, kernel_size=1, padding="same", name="conv2"
        )
        # Use shortcut conv on channel change or downsample.
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.conv_shortcut = which_conv(
                filters=self.out_ch,
                kernel_size=1,
                padding="same",
                name="conv_shortcut",
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
        self.skip_gain = self.add_weight(
            name="skip_gain",
            shape=(),
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, x, training):
        out = tf.keras.layers.Lambda(self.activation)(x) * self.beta
        if self.stride > 1:  # Average-pool downsample.
            shortcut = tf.keras.layers.AveragePooling2D(
                pool_size=(2, 2), strides=(2, 2), padding="same"
            )(out)
            if self.use_projection:
                shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x
        out = self.conv0(out)
        out = self.conv1(tf.keras.layers.Lambda(self.activation)(out))
        if self.use_two_convs:
            out = self.conv1b(tf.keras.layers.Lambda(self.activation)(out))
        out = self.conv2(tf.keras.layers.Lambda(self.activation)(out))
        out = (self.se(out) * 2) * out  # Multiply by 2 for rescaling
        # Get average residual standard deviation for reporting metrics.
        res_avg_var = tf.math.reduce_mean(
            tf.math.reduce_variance(out, axis=[0, 1, 2])
        )
        # Apply stochdepth if applicable.
        if self._has_stochdepth:
            out = self.stoch_depth(out, training)
        # SkipInit Gain
        out = out * self.skip_gain
        return out * self.alpha + shortcut, res_avg_var


def unitwise_norm(x):
    """Computes norms of each output unit separately, assuming (HW)IO weights."""
    if len(x.shape) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(x.shape) in [2, 3]:  # Linear layers of shape IO
        axis = 0
        keepdims = True
    elif len(x.shape) == 4:  # Conv kernels of shape HWIO
        axis = [
            0,
            1,
            2,
        ]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 3, 4]! {x}")
    return tf.math.reduce_sum(x ** 2, axis=axis, keepdims=keepdims) ** 0.5


def clip_gradient(grad, weight, clipping=0.01, eps=1e-3):
    param_norm = tf.math.maximum(unitwise_norm(weight), eps)
    grad_norm = unitwise_norm(grad)
    max_norm = param_norm * clipping
    # If grad norm > clipping * param_norm, rescale
    trigger = grad_norm > max_norm
    # Note the max(||G||, 1e-6) is technically unnecessary here, as
    # the clipping shouldn't trigger if the grad norm is zero,
    # but we include it in practice as a "just-in-case".
    clipped_grad = grad * (max_norm / tf.math.maximum(grad_norm, 1e-6))
    return tf.where(trigger, clipped_grad, grad)