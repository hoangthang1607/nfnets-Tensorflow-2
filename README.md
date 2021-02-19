# Tensorflow implementation of Normalizer-Free Networks and SGD - Adaptive Gradient Clipping

Paper: https://arxiv.org/abs/2102.06171.pdf

Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

# Installation and Usage

I recommend using Docker to run the code:

`docker build -t nfnets/imagenet:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .`

To train NFNets on imagenet dataset:

```
docker run --rm -it --gpus all -v $(pwd):/tf -p 8889:8888 -p 6006:6006 nfnets/imagenet:latest python train.py --variant F0 --batch_size 4096 --num_epochs 360
```

Please see the `train.py` module to get more arguments.

Pre-trained weights have been converted to be compatible with my models' implementation. You can download them from [here](https://drive.google.com/drive/folders/1HOd1BCFHPYHQMg6nh0DXH0b9_S9aMuTM?usp=sharing)

To evaluate NFNets on test set of imagenet dataset:

```
docker run --rm -it --gpus all -v $(pwd):/tf -p 8889:8888 -p 6006:6006 nfnets/imagenet:latest python evaluate_imagenet.py --variant F0 --batch_size 50
```

You can also check the notebook in the repo showing how to run an NFNet to classify an image.

# TODO
- [x] WSConv2d
- [x] Clipping Gradient module
- [ ] Documentation
- [x] NFNets
- [ ] NF-ResNets
- [ ] Update pretrained weights
- [ ] How to find-tune

# Cite Original Work

To cite the original paper, use:
```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```