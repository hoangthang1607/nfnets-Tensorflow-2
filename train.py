import argparse
from typing import Callable

import tensorflow as tf
import tensorflow_addons as tfa

from dataset import load, Split
from nfnet import NFNet, nfnet_params


NUM_CLASSES = 1000
NUM_IMAGES = 1281167


def parse_args():
    """Parse command line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-v",
        "--variant",
        default="F0",
        type=str,
        help="model variant",
    )
    ap.add_argument(
        "-b",
        "--batch_size",
        default=4096,
        type=int,
        help="train batch size",
    )
    ap.add_argument(
        "-n",
        "--num_epochs",
        default=360,
        type=int,
        help="number of training epochs",
    )
    ap.add_argument(
        "-l",
        "--label_smoothing",
        default=0.1,
        type=float,
        help="label_smoothing",
    )
    ap.add_argument(
        "-lr",
        "--learning_rate",
        default=0.1,
        type=float,
        help="learning rate",
    )
    ap.add_argument(
        "-d",
        "--drop_rate",
        default=0.2,
        type=float,
        help="drop rate",
    )
    ap.add_argument(
        "-e",
        "--ema_decay",
        default=0.99999,
        type=float,
        help="ema_decay",
    )
    ap.add_argument(
        "-c",
        "--clipping",
        default=0.01,
        type=float,
        help="AGC clipping param",
    )
    return ap.parse_args()


def main(args):
    steps_per_epoch = NUM_IMAGES // args.batch_size
    training_steps = (NUM_IMAGES * args.num_epochs) // args.batch_size
    train_imsize = nfnet_params[args.variant]["train_imsize"]
    test_imsize = nfnet_params[args.variant]["test_imsize"]
    aug_base_name = "cutmix_mixup_randaugment"
    augment_name = f"{aug_base_name}_{nfnet_params[args.variant]['RA_level']}"
    max_lr = args.learning_rate * args.batch_size / 256
    eval_preproc = "resize_crop_32"

    model = NFNet(
        num_classes=1000,
        variant=args.variant,
        drop_rate=args.drop_rate,
        label_smoothing=args.label_smoothing,
        ema_decay=args.ema_decay,
        clipping_factor=args.clipping
    )
    model.build((1, train_imsize, train_imsize, 3))
    lr_decayed_fn = tf.keras.experimental.CosineDecay(
        initial_learning_rate=max_lr,
        decay_steps=training_steps - 5 * steps_per_epoch,
    )
    lr_schedule = WarmUp(
        initial_learning_rate=max_lr,
        decay_schedule_fn=lr_decayed_fn,
        warmup_steps=5 * steps_per_epoch,
    )
    optimizer = tfa.optimizers.SGDW(
        learning_rate=lr_schedule, weight_decay=2e-5, momentum=0.9
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top_1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top_5_acc"
            ),
        ],
    )
    ds_train = load(
        Split(2),
        is_training=True,
        batch_dims=(args.batch_size,),  # dtype=tf.bfloat16,
        image_size=(train_imsize, train_imsize),
        augment_name=augment_name,
    )
    # ds_valid = load(Split(3), is_training=False, batch_dims=(256, ), augment_name="cutmix")
    ds_test = load(
        Split(4),
        is_training=False,
        batch_dims=(25,),  # dtype=tf.bfloat16,
        image_size=(test_imsize, test_imsize),
        eval_preproc=eval_preproc,
    )
    model.fit(
        ds_train,
        validation_data=ds_test,
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tf.keras.callbacks.TensorBoard()],
    )


# Patched from: https://huggingface.co/transformers/_modules/transformers/optimization_tf.html#WarmUp
class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (:obj:`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (:obj:`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (:obj:`int`):
            The number of steps for the warmup part of training.
        power (:obj:`float`, `optional`, defaults to 1):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (:obj:`str`, `optional`):
            Optional name prefix for the returned tensors during the schedule.
    """

    def __init__(
        self,
        initial_learning_rate: float,
        decay_schedule_fn: Callable,
        warmup_steps: int,
        power: float = 1.0,
        name: str = None,
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.power = power
        self.decay_schedule_fn = decay_schedule_fn
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "WarmUp") as name:
            # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
            # learning rate will be `global_step/num_warmup_steps * init_lr`.
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
                warmup_percent_done, self.power
            )
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.decay_schedule_fn(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_schedule_fn": self.decay_schedule_fn,
            "warmup_steps": self.warmup_steps,
            "power": self.power,
            "name": self.name,
        }


if __name__ == "__main__":
    args = parse_args()
    main(args)