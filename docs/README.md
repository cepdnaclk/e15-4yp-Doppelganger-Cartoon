---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title: Doppelganger Cartoon
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# Project Title

#### Team

- E/15/065, De Silva K.G.P.M., [e15065@eng.pdn.ac.lk](mailto:e15065@eng.pdn.ac.lk)
- E/15/076, Dileka J.H.S., [e15076@eng.pdn.ac.lk](mailto:e15076@eng.pdn.ac.lk)
- E/15/220, Maliththa K.H.H., [e15220@eng.pdn.ac.lk](mailto:e15220@eng.pdn.ac.lk)

#### Supervisors

- Dr. Asitha Bandaranayake, [asithab@pdn.ac.lk](mailto:asithab@pdn.ac.lk)
- Mr. Sampath Deegalla, [dsdeegalla@pdn.ac.lk](mailto:dsdeegalla@pdn.ac.lk)
- Mr. Ishan Gammampila

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)

---

## Abstract

Human face recognition and feature extraction have been the most interesting technologies to study for many researchers. It allows a huge number of face images to be recognized in just a short amount of time and extract the face features very easily, rather than recognizing each image and it's features individually through a normal human's eyes.Using these technologies researches are being carried out to find the look-alike characters within humans. Using methods for real people, cartoon character faces can hardly be detected and recognized because the face features of cartoon characters differ greatly from those of real people in terms of size and shape. This research was conduct to find the techniques to face detection,feature extraction of a cartoon characters and recognize look-alike cartoon character for a given human image. We have created Disney cartoon repository including 800 images from 77 characters, 5 images from each character with mirror images. Also include face features marked by hand as 35 labeled coordinates. For cartoon face detection and feature extraction, landmark based model trained using feature marked dataset. Used distances and the areas between the landmarks as features. Total 92 features(50 areas and 42 distances) are stored as csv files along with the cartoon images. To compare features of a real image with all the cartoon image features euclidean distance was considered. To increase the accuracy we used landmark based model with hair extraction model and also include gender prediction model. This combined model improves the performance compared to basic landmark based model. Alternatively, we implemented a classification model to find the best matching cartoon character. It shows 84\% accuracy on training data and 80\% accuracy on validation after 100 epochs. Finally we were able to find the best matching Doppelganger Cartoon character with good accuracy. So we hope this research and the dataset created by us will be more useful to other researchers.

## Related works
There is not any directly related project happening to find a machine learning algorithm that can find the cartoon character that best looks like you. But some websites published manually founded cartoon images with their matching human image
[10][11][13][14]. Also, some websites provide the best matching celebrity image for your
uploaded image.[12]Chase, Davis, and Amanda Jacquez did the research and reported
it called Finding Your Celebrity Look Alike.[15] They found vector representation of
faces and then used OpenCV HaarCascade classifier in order to detect faces. Images
inputted to the system and images in IMDB-WIKI data set were represented as 2622
dimensional vectors. Then IMDB-WIKI data set to compare with the inputted image
and find the similarities between them. According to that, they find the best matching
celebrity image/images using the Euclidean distance.

### Similarity Learning
A similarity measure is defined as the distance between various data points. Measuring
the similarity between two images is mostly used in image retrieval and computer vision
Fields.
SimNet[16] is a methodology proposed by Srikar Appalaraju and Vineet Chaoji,
This deep siamese network is trained on pairs of positive and negative images using a
novel online pair mining strategy inspired by Curriculum learning. Wang, J., Song, Y.,
Leung, T., Rosenberg, C., Wang, J., Philbin, J., Chen, B. and Wu, Y. proposed a deep
ranking model that learns the similarity metrics directly from images[17]. By comparing
the models which are based on handcrafted features, this approach has higher learning
capability. Their goal was to learn similarity models. Euclidean distance and nearest
neighbor search problem concepts were used to rank similar images. For ranking the loss function, a triplet based network was proposed and image triplets were taken as
the inputs. Because of the recent success of the ConvNet for image classification[18],
they started a convolution network that contains convolution layers, max-pooling layers,
local normalization layers, and fully-connected layers for each individual network. An
asynchronized stochastic gradient algorithm[19] with a momentum algorithm[20] was
used because training a deep neural network needs a large amount of data. The ImageNet
ILSVRC-2012 data set was used as training data which contains roughly 1000 images in
each of 1000 categories. A relevance training data which was generated in a bootstrap
fashion was also used as a training data set.

### Face Detection & Recognition
The computer technology that finds and identifies the human faces in digital images
is called face detection. Feature -based approach and image-based approach are the
two main approaches of detecting faces of real people[8]. Imager:: Anime Face is an
image-based approach which is used to detect faces of cartoon images[21]. This method
finds that the input images are faces or non-faces. As face detection, face recognition
also can be classified into two categories called model-based approach and image-based
approach. Kohei Takayama, Henry Johan and Tomoyuki Nishita proposed the first
relevant work concerning face detection and face recognition of cartoon characters
extracting the features[22]. In face detection the skin and the edges of the input image
were extracted firstly. Edges were extracted using Canny Method and they considered
that the skin color of the cartoon image has to be near to real people. Jaw contour and
Face symmetry are used to face detection. For comparison, OpenCV Face Detection
and Imager:: AnimeFace[21] are used as candidates and 493 various cartoon characters
are given as inputs. By comparing the results, the proposed method is more accurate
than the previous method. Feature extraction of the detected face and determination of
the individuality of the face and Character search are the main two purposes of their
face recognition system. Skin color, Hair color and the Hair quantity are the three
features that they extracted to build the feature vector. Face similarity is calculated
by measuring the distance between the features of two feature vectors of input image
and images in the database. 71% of output images contain the same characters as input
images(success) and 29% of the search are failure. Saurav Jha, Nikhil Agarwal and
Suneeta Agrawal presented a methodology to improve the Cartoon Face Detection and
Recognition systems. MTCNN architecture which offers a deep cascaded multi-task
framework is used to face detection. This architecture has three sequential deep CNNs and they are Proposal Net, Residual Net and the Output Net. For securing the baseline
results, Haar Features and HOGfeature are employed. Face recognition is experimented
on two different techniques, inductive transfer using inception v3 + SVM/GB and their
proposed method. The proposed method has two phases. The first phase consists of
preprocessing ( converting the cartoon image to gray scale and normalized), landmark
extraction(15 facial landmarks of 750 images of 50 characters) and landmark detection(5
layer LetNet architecture). In phase 2, leverages the images using a hybrid CNN (HCNN)
model. Benchmark IIIT-CFW database which contains 8,928 annotated cartoon images,
is used as the dataset.

### Feature Extraction
Feature extraction is important when finding a similar face to another face such as face
recognition, face detection, and expression detection. Eyes, mouth, and nose are the most
important features for face recognition[23]. Hua Gu, Guangda Su, and Cheng Du from
Tsinghua University proposed a method to extract feature points from faces[3]. This
approach is based on human visual characteristics. The features of the face are extracted
with the properties by using the geometry and the symmetry of faces. Normalizing the
image size before processing is not needed in this method. Integrating the local edge
information is not easy when we extract the face features. In this method, the Smallest
Univalue Segment Assimilating Nucleus(SUSAN) operator was chosen to extract the
edge and corner points of the feature area. Feature points were located by using face
similarity and geometry.
Lilipta Kumar Bhatta and Debaraj Rana proposed a technique for extracting facial
features from a color image through skin region extraction. Extracting the characteristics
of human face color and face region using Sobel operator[24], Converting the image
into YCbCr components and extracting skin region using morphological operation, and
extracting the regions of the human eye, mouth, and nose by means of gray level intensity
value were the three steps of their proposed technique. FEI face database[25] was used
for their experiment. They normalized the image size to 640*480. Using this technique,
they experimented and showed that the locating of the feature points is exact and fast,
this technique increases the accuracy of face recognition.

### Face Landmark Detection
Facial landmarks detection is used in many computer vision applications like face alignment, drowsiness detection, face recognition, facial expression analysis, facial animation,
3D face reconstruction as well as facial beautification, etc.[26] The aim of face landmark
detection is to detect the predefined key points like eyes, eyebrows, mouth, nose, etc. Yue
Wu·Qiang Ji classified these detection algorithms into three methods like holistic methods, Constrained Local Model (CLM) methods, and regression-based methods depending
on how they model the facial appearance and facial shape patterns. The holistic methods
models represent the global facial appearance and shape information. The Constrained
Local Model leverages the global shape model but builds the local appearance models.
And the regression-based methods capture facial shape and appearance information. [27]
Yongzhe Yan1,Xavier Naturel,Thierry Chateau, Stefan Duffner, Christophe Garcia,
Christophe Blanc divided facial landmark detection algorithms mainly into two types,
generative or discriminative. The generative types algorithms, which include the partbased generative models like ASM and holistic generative models like AAM, model the
facial shape and facial appearance as probabilistic distributions. They have provided
a comparison of different face alignment methods as well as different deep compression
models. To this comparison, they included traditional cascaded regression methods and
deep learning-based face alignment methods.[26]
Zixuan Xu1, Banghuai Li2, Miao Geng3, Ye Yuan identified that face landmarks
detection becomes a challenging task when dealing with faces in unconstrained scenarios,
especially with large pose variations. They targeted the problem of facial landmark
localization across large poses and give a solution based on a split-and-aggregate strategy.
When splitting the search space, they proposed a set of anchor templates as references for
regression, which well solved the problem which had with large variations of face poses.
Then depending on the prediction of each anchor template, they proposed to aggregate
the results, which reduce the landmark uncertainty due to the large poses.[28]

### Hair Segmentation
Since the appearance of hair can vary between different people based on their gender, age,
ethnicity, and the surrounding environment, automatic hair segmentation is challenging
in general.
Recently, there has been much success with deep neural networks (DNNs) and in many
tasks, including semantic segmentation, DNN-based hair segmentation methods havebeen introduced. The work of Liuet al. [29] introduced a multi-objective learning method
for deep convolutional networks that jointly models pixel-wise likelihoods and label
dependencies. A nonparametric prior was used for additional regularization, resulting in
better performance. Guo and Aarabi [30] presented a method for binary classification
using neural networks that perform training and classification on the same data using the
help of a pre-training heuristic classifier. They used a heuristic method to mine positive
and negative hair patches from each image with high confidence and trained a separate
DNN for each image, which was then used to classify the remaining pixels.

## Methodology
### Proposed Methodology
A machine learning algorithm to find the doppelganger cartoon for a given image is the
final outcome of this research. After reviewing previous works on face detection, feature
extraction and feature comparison, the proposed methodology is under the following
conditions.
• Cartoon images are limited to only Disney characters.
• Full body of cartoon images and real human images are not compared.
• Real human images should be given as the input.

### Conceptual design
There are more approaches done to detect faces, extract features and measure similarity of
images of real images and cartoon images separately. But there are very few applications
that compare cartoons and real humans using these concepts. After an extensive
study of the work done by various approaches and experiments, we came up with a
methodology. This application provides a number of analysis steps including preprocessing,
face detection, feature extraction, measuring similarity and displaying the results with a
user friendly web application. The web application is designed for users who want to
find the doppelganger of him/her. The real image is obtained and the result is displayed
through the web application. Image preprocessing, face detection, feature extraction and
measuring similarity steps are done in the backend.

![alt text](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Doc_Images/big_picature_of_methodology.PNG)

###  Web Application
Frontend of the web application is designed using React and the backend is developed
using python Django. React is an efficient, flexible javascript library which is developed
by Facebook for building interactive web applications. It lets us compose complex user
interfaces from small pieces of codes.The web application basically provides two features.
Users can upload a photograph of a person and see the resulting cartoon image. And
also they can share it to social media sites like facebook, instagram etc. Django REST
framework is a powerful and flexible toolkit for building Web APIs. Since we are building
our machine learning model using python tensorflow and keras libraries having a python
based backend is easy.


![alt text](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Doc_Images/backend_process_overview.PNG)

![alt text](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Doc_Images/overall_pipeline_of_method.PNG)

### Methodological approach
#### Data Collection
For image classification tasks there are some popular data sets that are used across research
and industry applications. The most popular ones are Imagenet, CIFAR, MINST. But for
tasks like cartoon-human image similarity checking there is no well-known dataset that
can directly be used. There are some freely accessible comic and animated cartoon image
repositories which differ more from human faces out there. Cartoon image repository in
this research is only contained with Disney cartoon images which are more similar to
human faces to simpler this approach as this is the beginning. Disney cartoon repository,
created by our own and ibug 300-w Human repositories are used to train the landmarks
detection model. For the classification model, the dataset contains 58 Disney cartoon characters with 406 images, 348 images for training, and 58 images for validation. To
test each algorithm of predicting the doppelganger cartoon image, we used a test set
contains 20 already known doppelgangers.

#### Data Preprocessing
Normalizing the images before feeding them into models is caused to give good results
and specific sizes are required from most models. Image data normalization ensures that
each pixel has a similar data distribution. This causes us to speed up the converging
process. Data normalization can be done by subtracting the mean from each pixel value
and dividing them by the standard deviation. So the normalized data is in the range of
[0,1] or [0,255].
As we were only considering 35 special landmarks on the face, we extracted that 35
landmarks from the given 68 landmarks in the human dataset ibug 300-W dataset. The
special 35 landmarks, considered in this research, is shown in the Figure 3.4.

#### Face detection and Feature extraction
After preprocessing images the next step is to extract features. This is the most important
part of this project because the accuracy of the algorithm directly depends on the extracted
features. Face detection and feature extraction can be done by various approaches. These
approaches are discussed in the section Face detection and Feature extraction

#### Store extracted features
When comparing images going through all the images in the repository, extracting
features and checking similarities will take a lot of time and need considerably huge
performance. So to reduce the effect of above problems we stored the extracted features
of cartoon images in a csv file with the image paths. So it is easy to go through the csv
file and check similarities with features extracted from human images.

#### Similarity
Distance metric or matching criteria is the main tool for finding the similar images. Two
vectors, a vector with extracted features of the real human image and a feature vector of a cartoon should be compared to find the similarity of the two images. The L1 metric
(Manhattan Distance), the L2 metric (Euclidean Distance) which are main two distance
metrics, have been proposed in the literature for measuring similarity between feature
vectors.
Euclidean Distance : If there is two points a and b have n dimensions such as
a=(x1 ,x2,...,xn) and b=(y1,y2,...,yn) , the Euclidean distance between two points can be
generalized as in Equation 3.1

The calculated Euclidean distances of each cartoon feature vector with the real image
feature vector are compared and the cartoon image with the least distance is selected as
the best matching cartoon image for the real image.

## Experiment Setup and Implementation
### Research Tools
In the purpose of implementing our project we have used several libraries and frameworks.
• Numpy : This is a library for python programming which supports multidimensional arrays and matrices, with a large collection of high level mathematical
functions.
• Keras : This is an API designed to follow best practices for reducing cognitive
loads and it offers consistent and simple APIs which minimizes the number of user
actions for common use cases.
• Tensorflow : Tensorflow is a open source library for machine learning. It can be
used across a range of tasks which involve deep neural networks.
• Cv2 : OpenCV is a library which is designed for solving computer vision problems.
• MobileNetV1: A family of general purpose computer vision neural networks
designed with mobile devices in mind to support classification, detection and more.
This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M
images and 1000 classes.
• Matplotlib : Matplotlib is a comprehensive library for creating static, animated,
and interactive visualizations in Python. In our case it is very useful in displaying
images.
• os: This module provides a portable way of using operating system dependent
functionality
• Csv: This library helps to manipulate csv files writing and reading.
• PIL: Pillow library adds fairly powerful image processing capabilities and provides
extensive file format support, and efficient internal representation.
• React: This is an open source front end development javascript library for building
interactive user interfaces.
• Django REST framework: This is a powerful and flexible toolkit for building
restful web APIs.
To run and test codes which are written in python, we used Google Colaboratory.
It is a jupyter notebook which runs in the cloud and is integrated with google drive,
making it easy to set up ,access and share. So the image repository is located in the
shared google drive. By default this notebook runs on the CPU. But it supports GPU
and TPU hardware acceleration for achieving higher performance.

### Data manipulation and Testing
Our cartoon data sets is a repository containing Disney cartoon images. Though the
pretrained models are not perfectly detecting the cartoon faces, this dataset is containing
only frontal faces of cartoon images. Data manipulation for cartoon landmarks detection
model is done by annotating the landmarks on the cartoon faces using a tool. The iBUG
300-W dataset which has already annotated landmarks is used as the human dataset to
train the model for landmarks detection.

### Pitfalls and workarounds
During this project, one of the main challenges encountered was to get the background
knowledge of the data set, feature extraction and face detection of images. As a remedy
for that issue, we had to do lots of background research. Finding a strong data set with
cartoon images is required for our task. At first, we could not properly understand the
already existing data sets. After gaining knowledge about previous works, we understood
that there are a number of data sets which contain various cartoon images but existing
data sets are still lacking similarities compared to humans. As a solution, we decided to
collect images to build a Disney cartoon image repository on our own as Disney cartoons

are more similar to human images. But still the lack of data for training purposes has
remained. A number of researches are done on face detection and feature extraction.
But one issue with that was lack of documentation about feature extraction and face
detection of cartoon images. After reviewing the research papers, we had to spend a
considerable amount of time figuring out how the pretrained models work on cartoon
images and human images to select a best model for our case. By analyzing the results
given by the models, we figured out that the pretrained models are not performed well
in cartoon face detection. But actually training a model from scratch is very expensive
and requires huge data sets to achieve good generalization performance.
The method landmarks detection, is required to detect the face first and detecting
face of cartoon images is not similar to human face detection. We handled this by using
the frontal face of cartoons and using the dlib frontal face detector as the detector.
Also, when running the code on google colabs, due to the high throughput of the data
set, we faced an issue of insufficient memory and low speed. We handled this problem by
switching the run time mode to GPU from none in the Google Colabs environment.

## Results and Analysis
### Overall analysis
All the algorithms we tried on this research are tested on the same test set which contains
already known doppelgangers. To compare the models we ranked the resulted images
according to the ascending order of euclidean distance. Table 5.2 Shows the summary of ranks for each cartoon. For some images, the models do not give an output because of
some failures in the hair extraction model or gender prediction model for some cases. As
an example, some female images can have short hair, then the gender prediction model
wrongly predicted the gender and then our expected result is not within the resulted
images.
Figure 5.32 shows the variation of the ranks of each cartoon character for different
weight values. By analyzing the graph, we can conclude that the rank varies for different
weights. Some cartoons get better rank in w = 0.5 and someones get better at w=0.2.
The graph of w=1 (landmarks-based model only) always gives higher ranks (far away
from expected result) for all cartoons with respect to other models. According to the
graph, w=0.2 and w=0.5 (combined model) gives some good ranks for all cartoons.
But according to the classification results in the Table 5.2, the classification model
gives the best results for all cartoons as all the expected outputs are within the top 5
ranks.

![alt text](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Doc_Images/summary%20table.PNG)


![alt text](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Doc_Images/Analysis.PNG)

## Conclusion
During recent years, many researches have been carried out in various ways of feature
extraction of human images, finding looks-alike twins, and so on. We entered the research
using the pre-trained model based approach and after analyzing the results, we concluded
that we should simplify this by considering only the Disney cartoons which are more
similar to humans as this is the beginning and train our models for building the algorithm.
In this paper, we mainly researched an approach that finds the best matching Cartoon
character for human image based on landmarks model. Because lack of existing cartoon
datasets, we have created a dataset with landmarks on the faces of cartoon characters
for training a model and it will be more useful for future researchers. Combination of
landmark based model with hair extraction model and gender prediction model has
improved the performance. But the best cartoon image is resulted for different weights
on the models for various images. Alternatively implemented classification model shows
84% accuracy on training data and 80% accuracy on validation after 100 epochs. As
features on cartoon faces such as eyes, nose are more differ from human features, the
combined model is also not accurate like classification. So, the classification algorithm
with a strong dataset will be the best model for this finding doppelganger task. Today’s
society is interesting to compare their appearance with cartoons because it brings mental
relaxation and fun for their minds. So we hope this research will be helpful for them.

## Publications
1. [Semester 7 report](https://drive.google.com/drive/u/2/folders/1Dla8D2qBbewU4_VNNaYW0Og9luAq-1RQ)
2. [Semester 7 slides](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Publications/CO%20421%20Final%20Presentation%20.pptx)
3. [Semester 8 report](https://drive.google.com/drive/u/2/folders/1Dla8D2qBbewU4_VNNaYW0Og9luAq-1RQ)
4. [Semester 8 slides](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon/blob/main/docs/Publications/CO%20425%20-final%20presentation.pptx)
5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./).


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e15-4yp-Doppelganger-Cartoon)
- [Project Page](https://cepdnaclk.github.io/e15-4yp-Doppelganger-Cartoon)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
