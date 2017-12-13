#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Report/Model_Arch.PNG "Model Visualization"
[image2]: ./Report/center_Straight.jpg "CenterImage"
[image3]: ./Report/center_right.jpg "Recovery Image"
[image4]: ./Report/center_left.jpg "Recovery Image"
[image5]: ./Report/center_2017_11_10_14_19_54_143.jpg "Recovery Image"
[image6]: ./Report/center-2017-02-06-16-20-04-855.jpg "Normal Image"
[image7]: ./Report/center-2017-02-06-16-20-04-855-flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* Resubmition.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The Resubmition.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
My initial approach was to use LeNet, but it was hard to have the car inside the street with differenet values of epochs.
After that, I decided to try the NVIDIA Architecture and the car drove the complete first track after just three training epochs.

NVIDIA Architecture consists of :
* 5 convolutional layers using relu function for activation
* 4 Fully connected layers


####2. Attempts to reduce overfitting in the model

 I started by high ephocs but I found that it had caused overfitting so I started to reduce the number of ephocs to 3. In addition to that, I split my sample data into training and validation data(80% as training , 20% as validation) .
####3. Model parameter tuning

The model used an adam optimizer with mse, so the learning rate was not tuned manually.

####4. Appropriate training data
 * I recorded data from first track using sim tool.
 * I extracted from these data Steering values and center image.
 * From the previous data , I composed new data set consists of:
    * Center image versus steering value 
    * Right image versus steering value + correction value (0.2)
    * Left image versus steering value - correction value (0.2)
    * flipped images and corresponding steering

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to reach good results to keep car moving in the middle of the track and apply the correct steering value.

My first step was to use a convolution neural network model similar to LENET I thought this model might be appropriate as I tried it before in detection traffic signs images and it works well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 
This implied that the model was overfitting. 

To combat the overfitting, I changed the model and used NVIDIA model.

Then I found it overfiting again with 5 ephocs , so I reduced it to 3. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I used the way mentioned in "Appropriate training data" section.
then I applied preprocessing on images :
* Lambda layer was used for normalization
* Images was cropped as well.


At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


After the collection process, I had 18966 * 2 number of imags. I then preprocessed this data by :
* Lambda layer was used for normalization
* Images was cropped as well.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as if I increased that number I had overfitting issue. I used an adam optimizer so that manually training the learning rate wasn't necessary.
