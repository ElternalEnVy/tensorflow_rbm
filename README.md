# tensorflow_rbm

#### Description
This is a tensorflow based implementation of restricted boltzmann machines.

#### Code
tfrbm file contains the code to build,train and evaluate RBM.

tfrbm->rbm.py define BBRBM and GBRBM

tfrbm->base_rbm.py define basic RBM

tfrbm->rbm_ais.py AIS implementation

tfrbm->utils.py contains tools tfrbm->plot_utils.py contains plot tools 

tfrbm->dataset.py process datasets

{dataset}_rbm.py means the build and training on {dataset}.

{dataset}_trprpr.py means the pruning experiment on {dataset}.

Note that caltech means CalTech 101 Silhouettes 16$\times$16 dataset and caltech0 means CalTech 101 Silhouettes 28$\times$28 dataset.

#### Instructions
For example for training RBM on MNIST.Run 

`python mnist_rbm.py --algorithm 'CD' --n-gibbs-steps 1 --anneal-lr --lr 0.05 --save-path='/documents/code/experiments/pruning_rbm/mnist/cd-25-500/' --n-hidden 500 --epochs 249`

For pruning experiment on MNIST.Run

`python mnist_trprtr.py --algorithm 'CD' --n-gibbs-steps 1 --lr 0.05 --anneal-lr --epochs 249  --save-path='/documents/code/experiments/pruning_rbm/mnist/cd-25-500/'`

Change the save_path as you like
