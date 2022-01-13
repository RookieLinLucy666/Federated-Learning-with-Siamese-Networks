# Federated Alzheimers Classification on MRIs
Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink (atrophy) and brain cells to die. Alzheimer's disease is the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently. The early signs of the disease include forgetting recent events or conversations. As the disease progresses, a person with Alzheimer's disease will develop severe memory impairment and lose the ability to carry out everyday tasks. The objective is to detect the various levels of Alzheimer's present in a patient using MRI scans. 

## File Structure
<br>`model.py` - Contains the model class (PyTorch)
<br>`utils.py` - Contains helper functions for loading datasets, training and validation
<br>`client.py` - Contains the client side code for Federated Learning 
<br>`server.py` - Contains the server side code for Federated Learning

## Usage

1. Open the terminal and navigate to the current directory
2. Run `python server.py`
3. Open a new terminal window and repeat Step 1
4. Run `python client.py`

Thus, we have 1  client and 1 server to run our project. To add more clients, simply open more terminal windows and repeat steps 3 and 4.


## Dataset

The dataset used is the [Alzheimer's Dataset (4 Class of Images)](https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images) from Kaggle. The 4 class labels are:

1.**Mild Demented**

2.**Moderate Demented**

3.**Non Demented**

4.**Very Mild Demented**



## Client Model Architecture - Convolutional Siamese Networks

The network architecture used was a basic CNN model, with Max Pooling and ReLU Activation functions. Input images are resized to an optimal size and then fed into the **Convolutional layer**. These images are converted to their pixel values, which can be imagined as a three-dimensional matrix for the purpose of visualization. The **Convolutional layer** has a kernel. This kernel is generally a small matrix of specified kernel size mxnx3 (3 for RGB images). 
<br>

**Rectified Linear Unit (ReLU)** is the activation layer used in CNNs.The activation function is applied to increase non-linearity in the CNN. Images are made of different objects that are not linear to each other.


**Max Pooling:** A limitation of the feature map output of Convolutional Layers is that they record the precise position of features in the input. This means that small movements in the position of the feature in the input image will result in a different feature map. This can happen with re-cropping, rotation, shifting, and other minor changes to the input image. A common approach to addressing this problem from signal processing is called down sampling. This is where a lower resolution version of an input signal is created that still contains the large or important structural elements, without the fine detail that may not be as useful to the task.

## Future Work

Transfer Learning based approaches using ResNets or AlexNet can be experimented with to see what sort of results are obtained.