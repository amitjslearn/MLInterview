# Convolutional Neural Network
The AlexNet, was first able to beat more traditional geometrical approaches on the most popular object recognition contest - the ILSVRC. Ever since, DCNNs have been achieving state-of-the-art results on object recognition tasks.

# Why CNN

If you would use Fully connected ANN then you would have to train a lot of parameters and network has to optimize all of these parameters,
the training process could then become very time and storage intensive. In order to solve this computational problem, a different
kind of network architecture is used, called **Convolutional Neural Network (CNN)**

CNNs introduce two new types of hidden layers in which each neuron is only connected to a small subset of the neurons in the previous layer, to prevent the
aforementioned problem.</br>

1. **Convolutional Layer**: What separates a convolutional layer from a fully-connected one is that each neuron is only connected to a small, local subset of the
neurons in the previous layer, which is a square sized region across the height and width dimensions. The size of this square
is a hyperparameter named **Receptive Field**.

![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2008-21-45.png)

Neurons of the convolution operator can recognize certain local patterns of the previous layer’s output.In order to now reconize multiple different features within
one layer, it is required to have several Filters. How many convolutions are being conducted is defined by another hyperparameter, called Stride, which determines
how big the gap between two scanned regions is.

Why Padding ?? (Having Certain Width & Height + Cover the edges of images much better )

2. **Pooling Layer**: The third kind of layer, which has the purpose of decreasing the complexity of CNNs, is the Pooling
Layer. Similarly to the convolutional layer, neurons in the pooling layer are connected to a square sized region across
the width and height dimensions of the previous layer. The main difference between convolution and pooling is that a
pooling layer is not Parametrized.

* Max Pooling
* Average Pooling

![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2008-22-13.png)

## APPLICATIONS OF DCNN FOR COMPUTER VISION (OBJECT RECOGNITION TASKS)

Three object recognition tasks - classification, localization and detection - and how each of them can be tackled with DCNNs.

### Classification

The task of Image Classification describes the challenge of categorizing a given image into one of several classes. A
possible application of this could be the recognition of hand-written digits, where the input image is classified as one of the
ten classes.

![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2008-43-37.png)

### Localization

For Localization, the information about which category an image belongs to is already available and the task is to instead
figure out where exactly the object is located in the image. This location is typically specified by a two-dimensional bounding
box, which consists of four values that describe the location of two opposite couples of corners. Finding these four values is
the main challenge of localization and is commonly referred to as **Bounding Box Regression**.

![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2008-49-08.png)

To perform a localization task, we can use a similar architecture as the one we defined for classification. The only
thing that has to be modified is the final output layer, which can simply be replaced by another output layer that performs
the bounding box regression instead.

### Detection

In contrast to multi-class localization, the number of objects in a given image is not known prior to the execution, when performing Object Detection. In order to use DCNNs for such
object detection tasks, the architecture needs to be extended to handle the flexible amount of detections.

  * **R-CNN**: **Region-based Convolutional Neural Networks (R-CNNs)** are using Region Proposal Networks to only extract potentially interesting regions. These regions are called
Regions of Interest (ROIs). They are obtained by running a quick segmentation to spot blob-like structures.


  * **R-FCN**: These networks are structured very similarly to the faster R-CNN architecture, but instead of using fully-
connected modules to predict classes and bounding boxes for each ROI, R-FCNs use Position-Sensitive Convolutional Modules.

  * **YOLO** (You Only Look Once): In contrast with region proposal based techniques, Single-Shot detection architectures do not predict any
regions of interests, but instead, a fixed amount of detections on the image directly, which are then filtered to contain only
the actual detections. These networks do therefore have much faster execution times than region-based architectures but are
found to also have a lower detection accuracy.

  Note : Suitable for real time applications.
  ![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2009-28-06.png)

  * **Single Shot MultiBox Detectors (SSD)** : In order to handle the problems of YOLO that arise due to the fixed amount of predictions and cell sizes,
Single Shot MultiBox Detectors (SSD) have been developed, which predict detections of different scales and also make
predictions for multiple different aspect ratios. Therefore, SSD detectors can make finer predictions, leading to significantly
better results.

___
Some of the Advanced Computer Vision tasks include Segmentation, Pose Estimation, Image Captioning, Generating Images (Neural Style Transfer)

### Segmentation
Segmentation is the task of assiging each pixel in a image to a class, it's like drawing the boundry around the object for every object in a given image. <br/>

Types of image segmentation 
1) Semantic Segmentation
2) Instance Segmentation

![](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/Screenshot-from-2019-03-28-12-08-09.png)

### Pose Estimation (Human Pose Estimation)
Pose Estimation involves localization of human joints (also known as keypoints - elbows, wrists, etc) in images.

Essentially, it is a set of coordinates that can be connected to describe the pose of the person.

 Human Pose Estimation has some pretty cool applications and is heavily used in Action recognition, Animation, Gaming, etc.

![](https://miro.medium.com/max/1200/1*Q8sk8FwMIYuLoLmptsT-MA.png)
- **2D Pose Estimation** - Estimate a 2D pose (x,y) coordinates for each joint from a RGB image.
- **3D Pose Estimation** - Estimate a 3D pose (x,y,z) coordinates a RGB image.

There are many approaches, you can refer [here](https://nanonets.com/blog/human-pose-estimation-2d-guide/) for more information.

### GAN (Generative Adversarial Networks)

GANs are generative models: they create new data instances that resemble your training data.
For example, GANs can create images that look like photographs of human faces, even though the faces don't belong to any real person.

Let's split the word **Generative Adversarial Networks**  <br/>
**Generative** - implies generation  <br/>
**Adversarial** - implies conflict or opposition  <br/>
**Networks**- implies that it's composed of neural networks  <br/>
So overall in GAN there is a fight between networks to get better at their tasks
GAN has two parts:

- The **generator** is a neural network, which learns to generate plausible data. 
  + The generated instances become negative training examples for the discriminator.
- The **discriminator** is a neural network, which learns to distinguish the generator's fake data from real data. 
   + The discriminator penalizes the generator for producing implausible results.

The generator output is connected directly to the discriminator input. Through backpropagation, the discriminator's classification provides a signal that the generator uses to update its weights. Gradually the generator becomes better at generating data/images that a discriminator can't tell whether it's real or fake (generated by **generator**).

### Neural Style Transfer

### Image Captioning
Generating captions/text that describe the given image. It uses both Natural Language Processing and Computer Vision to generate the captions.
![](https://miro.medium.com/max/1400/1*6BFOIdSHlk24Z3DFEakvnQ.png)


## DCNN ARCHITECTURES

ImageNet Large Scale Visual Recognition Competition (ILSVRC) - Models in chronological order.

* **LeNet-5 (1998)** [3 CL + 2 Pooling + 1 FC]

Most of the DCNNs that are being used for object recognition today are based on this basic architecture. As can be seen, the network architecture is relatively simple,
as it only consists of an input layer of size 32 × 32, an output layer of size 10, as well as three 5 × 5 convolutional, two
2×2 pooling and one fully-connected layer in between, making it a total of six hidden layers.
![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2009-32-15.png)

* **AlexNet (2012)** [5 CL + 3 Pooling + 2 FC]

It was the first DCNN that managed to beat more traditional object recognition approaches in the ILSVRC.
![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2009-32-15.png)

* ZFNet and OverFeat (2013) [Same as AlexNet but with a slight modification]

* **VGGNet and GoogLeNet (2014)**

VGGNet, scored second place in the ILSVRC and influenced the deep learning scene in an important way, as they showed that using a deeper architecture
does generally lead to better results, which was not obvious at that time.
![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2009-45-48.png)

Google won the ILSVRC with a different very deep network that had 22 parametrized layers - even more than the VGGNet - and was named GoogLeNet. This network was an improvement of the AlexNet that was not only much deeper but also
reduced the number of parameters. The latter was achieved by replacing the first fully-connected layer, which is typically
accountable for the highest number of parameters, by another convolutional layer. In addition to that, they also implemented
the so-called Inception Modules, which enable a network to recognize patterns of different sizes within the same layer.
![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2010-00-46.png)

* **ResNet** (2015)

Their network, called Deep Residual Network, or ResNet for short, is able to perform much better with very deep architectures. In
ResNets, convolutional layers are divided into Residual Blocks and for each block a Residual Connections is added, which is
bypassing the corresponding block. By adding these residual connections, the result of a training
step can be backpropagated to the earlier layers directly, without any interference from subsequent layers. Therefore, resid-
ual connections enable the training of even deeper networks.

![](https://github.com/theainerd/MLInterview/blob/master/Deep%20Learning/images/Screenshot%20from%202018-09-30%2010-08-03.png)


* Inception-v4, Inception-ResNet-v1 and ResNeXt (2016)

* Densenet, DPN and MobileNets (2017)



## Refrences:

* [A Non-Technical Survey on Deep Convolutional Neural Network Architectures](https://arxiv.org/pdf/1803.02129.pdf):star::star::star::star:
* [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
* [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/)
* [Nanonets](https://nanonets.com/blog/human-pose-estimation-2d-guide/)
* [google developers](https://developers.google.com/machine-learning/gan/gan_structure)