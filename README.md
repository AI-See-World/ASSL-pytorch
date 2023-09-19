# ASSL-pytorch
This repository contains an implementation of ASSL in PyTorch, which can reproduce the results for CIFAR10 and Caltech256 datasets as reported in the paper.

The paper proposes the ASSL (Adversarial Sample Synthesis and Labeling) framework to address the problem of stealing black-box models when only input data and hard-label outputs are available. The framework utilizes a large pool of unlabeled samples to train a substitute model that accurately simulates the target black-box model. Experimental results demonstrate that the proposed method outperforms state-of-the-art techniques by 3.73% to 16.94% when faced with hard-label outputs.
![](assl\famework.png)

## Requirements
- python 3.6
- pytorch 1.5.10
- torchvision 0.10.0
- tensorboard 2.6.0
- pillow

## Install dependencies
You can install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

## How to Use to Train
To train the model, follow the steps below:

- Install PyTorch by selecting your environment on the website and running the appropriate command.
- Clone this repository.
- Note: We currently only support Python 3+.
- Modify the corresponding dataset location in ssl_dataset.py before training starts.
- Download the dataset by following the instructions provided in the repository.

Run the following command to start the training:
```
python train.py 
```
## Result
![](assl\table1.png)
![](assl\table2.png)

## Evaluate
To evaluate the trained model, you can use the eval.py script.

## Performance
We report the performance of the model on the CIFAR10 and Caltech256 datasets in the paper corresponding to the code.

## Future Work
Future work on this project may include additional improvements to the model architecture and exploring new datasets.



