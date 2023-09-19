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

The aim of this project is to obtain high-quality training data by downloading and processing images from the ILSVRC-2012 dataset and using a pre-trained Imagenet model for high-confidence filtering of the images. The following are the specific steps to accomplish this task using Python and PyTorch:
- Download the ILSVRC-2012 Dataset：Before starting, make sure you have downloaded the ILSVRC-2012 dataset, which contains 1.2 million images. You can obtain the dataset from a suitable source and save it in your local environment.
- Import Necessary Libraries：In your Python code, first import the required libraries, including PyTorch and torchvision.models. These libraries will be used to load the pre-trained Imagenet model and perform image preprocessing.
```
import torch
import torchvision.models as models
```
- Create and Load the Pre-trained Model：In this example, we choose to use ResNet-50 as the pre-trained model. You can choose other models according to your needs. The following code loads the ResNet-50 model and loads the pre-trained weights:
```
model = models.resnet50(pretrained=True)
```
- Image Preprocessing：Before feeding the images into the model, they need to be preprocessed. This includes resizing the images to a specified size, performing center cropping, converting them to tensor format, and normalizing them. The following code snippet demonstrates how to preprocess the images using torchvision.transforms:
```
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
- Set the Model to Evaluation Mode：
```
model.eval()
```
- Dataset Filtering and Selection:Next, we will perform inference on the entire dataset using the pre-trained model and select images with high-confidence predictions. To ensure dataset diversity, we will uniformly select a certain number of samples from each category in Imagenet. Specifically, we will sort each category based on the highest confidence output by the pre-trained model and select the top k samples with the highest confidence. The value of k is calculated as 122,186 (the number of categories in the ILSVRC-2012 dataset) divided by the number of categories in the Imagenet dataset being used.
- Code File Structure:After completing dataset filtering and selection, you can organize the relevant code into a single file for easy replacement of the dataset path and project execution. Our approach is to store the dataset categorized in different folders and in .jpg format.


## How to Use to Train
To train the model, follow the steps below:

- Clone the repository：Use Git or other version control tools to clone the repository to your local computer:
```
git clone <repository_url>
```
- Python version: Make sure your Python version is 3.x or above, as Python 3+ is currently supported.
- Modify the dataset path: Before starting the training, please modify lines 254, 257, and 262 in the ssl_dataset.py file to match the location of your dataset on the server. This ensures that the training process can load the dataset correctly.Please note the differences in the paths of the datasets we need to modify:
```
path_initial: This is the path to the dataset used for training the initial substitute model. The initial training dataset is sourced from AssData.
path: This is the path to the dataset from AssData.
```

- Prepare the black-box model: Before training the agent model, make sure you have prepared the black-box model. In this experiment, we assume that the black-box model is a ResNet34 model trained on the CIFAR-10 dataset. Please note that when saving the black-box model, save it in the form of a complete model + parameters. If only the model is saved, you will need to provide the network structure parameters when loading the black-box model to load it correctly. The code to load the black-box model is located at line 45 of the ssl_dataset.py file. Make sure to provide the correct network structure parameters when loading the black-box model.
- We have a [blackbox](https://github.com/sau-GaoLijun/ASSL-pytorch/tree/master) model of cifar10 already trained that you can put directly into your project.
- Important hyperparameter settings: In the main function of the train.py file, you can set the following important hyperparameters to fit your server and training requirements:
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



