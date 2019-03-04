# tensorflow_rbm

### Dependencies

### Description
This is a tensorflow based implementation of restricted boltzmann machines.
<br>
Also the code for pruning in RBM,Zhiwen Zuo et al.([https://arxiv.org/abs/1901.07066])
<br>

### Dataset
MNIST,OCR letters,NORB and CalTech 101 Silhouettes datasets can be downloaded by runing the shell scripts
in data folder.

### Code
```
.
├── README.md
└── tfrbm
    ├── base_rbm.py
    ├── dataset.py
    ├── plot_utils.py
    ├── rbm_ais.py
    ├── rbm.py
    └── utils.py
├──...

```
tfrbm folder contains the code to build,train and evaluate RBM.

>rbm.py define BBRBM and GBRBM

>base_rbm.py define basic RBM

>rbm_ais.py Anneal Importance Sampling(AIS) implementation

>utils.py and plot_utils.py contain tools 

>dataset.py process and import datasets

>{dataset}_rbm.py build and training on {dataset}.

>{dataset}_trprpr.py perform pruning experiment on {dataset}.

Note that {caltech} means CalTech 101 Silhouettes 16 $$\times$$ 16 dataset and {caltech0} means CalTech 101 Silhouettes 28 $$\times$$ 28 dataset.

### Instructions
For example for training RBM on MNIST dataset

```sh
python mnist_rbm.py --algorithm 'CD' --n-gibbs-steps 1 --anneal-lr --lr 0.05 --save-path='/documents/code/experiments/pruning_rbm/mnist/cd-25-500/' --n-hidden 500 --epochs 249
```

For pruning experiment on MNIST.Run

```sh
python mnist_trprtr.py --algorithm 'CD' --n-gibbs-steps 1 --lr 0.05 --anneal-lr --epochs 249  --save-path='/documents/code/experiments/pruning_rbm/mnist/cd-25-500/'
```

