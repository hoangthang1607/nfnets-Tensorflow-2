import argparse

from dataset import load, Split
from nfnet import NFNet, nfnet_params


NUM_CLASSES = 1000
NUM_IMAGES = 50000


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
        default=25,
        type=int,
        help="test batch size",
    )
    return ap.parse_args()


def main(args):
    steps_per_epoch = NUM_IMAGES // args.batch_size
    test_imsize = nfnet_params[args.variant]["test_imsize"]
    eval_preproc = "resize_crop_32"

    model = NFNet(
        num_classes=1000,
        variant=args.variant,
    )
    model.build((1, test_imsize, test_imsize, 3))
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="top_1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, name="top_5_acc"
            ),
        ],
    )
    ds_test = load(
        Split(4),
        is_training=False,
        batch_dims=(args.batch_size,),  # dtype=tf.bfloat16,
        image_size=(test_imsize, test_imsize),
        eval_preproc=eval_preproc,
    )
    model.load_weights(f"{args.variant}_NFNet/{args.variant}_NFNet")
    model.evaluate(ds_test, steps=steps_per_epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)