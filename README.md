# Tensorflow implementation of Normalizer-Free Networks and SGD - Adaptive Gradient Clipping

Paper: https://arxiv.org/abs/2102.06171.pdf

Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

# Installation and Usage

I recommend using Docker to run the code:

`docker build -t nfnets/imagenet:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .`

to train NFNets on imagenet dataset:

```
docker run --rm -it --gpus all -v $(pwd):/tf -p 8889:8888 -p 6006:6006 nfnets/imagenet:latest python train.py --variant F0 --batch_size 4096 --num_epochs 360
```

Please see the `train.py` module to get more arguments.

# TODO
- [x] WSConv2d
- [x] Clipping Gradient module
- [ ] Documentation
- [x] NFNets
- [ ] NF-ResNets

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