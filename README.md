# ASSL-pytorch
This repository contains an implementation of ASSL in PyTorch, which can reproduce the results for CIFAR10 and Caltech256 datasets as reported in the paper.

The paper proposes the ASSL (Adversarial Sample Synthesis and Labeling) framework to address the problem of stealing black-box models when only input data and hard-label outputs are available. The framework utilizes a large pool of unlabeled samples to train a substitute model that accurately simulates the target black-box model. Experimental results demonstrate that the proposed method outperforms state-of-the-art techniques by 3.73% to 16.94% when faced with hard-label outputs.
![](https://github.com/sau-GaoLijun/ASSL-pytorch/blob/main/assl-code/assl/famework.png)

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
## Prepare the train data 
You can prepare the training dataset as follows：https://github.com/sau-GaoLijun/ASSL-pytorch/blob/main/assl-code/train_data.md

## How to Use to Train
To train the model, follow the steps below:

- Clone the repository：Use Git or other version control tools to clone the repository to your local computer:
```
git clone <repository_url>
```
- Python version: Make sure your Python version is 3.x or above, as Python 3+ is currently supported.
- Modify the dataset path: Before starting the training, please modify lines 254, 257, and 262 in the ssl_dataset.py file to match the location of your dataset on the server. This ensures that the training process can load the dataset correctly.
- Prepare the black-box model: Before training the agent model, make sure you have prepared the black-box model. In this experiment, we assume that the black-box model is a ResNet34 model trained on the CIFAR-10 dataset. Please note that when saving the black-box model, save it in the form of a complete model + parameters. If only the model is saved, you will need to provide the network structure parameters when loading the black-box model to load it correctly. The code to load the black-box model is located at line 45 of the ssl_dataset.py file. Make sure to provide the correct network structure parameters when loading the black-box model.
- Important hyperparameter settings: In the main function of the train.py file, you can set the following important hyperparameters to fit your server and training requirements:
- 
```
--save_dir : Specify the directory where the model will be saved. The default is ./saved_models.

--save_name : Specify the filename of the saved model. The default is cifar10-40.

--resume : If set to True, resume training from the previous training session. The default is False.

--load_path : Specify the path of the pretrained model to load. The default is None.

--overwrite : If set to True, overwrite the existing model with the same name when saving the model. The default is True.

--epoch : Specify the total number of training epochs. The default is 10.

--num_train_iter : Specify the total number of training iterations (equal to epoch multiplied by every_num_train_iter).

--every_num_train_iter : Specify the number of training iterations per epoch.
```
 Now, you can run the following command to start the training process. Make sure you are in the root directory of the repository and execute the following command in the command line or terminal:
```
python train.py
```


## Result
<img src="https://github.com/sau-GaoLijun/ASSL-pytorch/blob/main/assl-code/assl/table1.png" width="310px"><img src="https://github.com/sau-GaoLijun/ASSL-pytorch/blob/main/assl-code/assl/table2.png" width="310px">



## Evaluate
To evaluate the trained model, you can use the eval.py script.

## Performance
We report the performance of the model on the CIFAR10 and Caltech256 datasets in the paper corresponding to the code.

## Future Work
Future work on this project may include additional improvements to the model architecture and exploring new datasets.



