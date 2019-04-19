# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_imgs/center.png "Center camera image"
[image2]: ./writeup_imgs/steering_distro.png "Distribution of steering angles"
[image3]: ./writeup_imgs/center_flip.png "Center camera image, flipped"
[image4]: ./writeup_imgs/center_crop.png "Center camera images, cropped"
[image5]: ./writeup_imgs/multi_cams.png "All camera images"
[image6]: ./writeup_imgs/multi_cams_flip.png "All camera images, flipped"
[image7]: ./writeup_imgs/multi_cams_crop.png "All camera images, cropped"
[image8]: ./writeup_imgs/multi_cams_flip_crop.png "All camera images, flipped and cropped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model.
* model.h5 containing a trained convolutional neural network .
* drive.py for driving the car in autonomous mode.
    * I did not modify this file.
* video.mp4 showing one lap around the test track.
* writeup_report.md summarizing the results. (This file!)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for creating training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths of 6, 16, and 32. Between convolution layers, there are max pooling layers with 2x2 filters and 2x2 strides. After the last max pooling layer, the units are flattened and fed into a feed-forward neural network with hidden layers containing 1000, 100, and 10 units. The final output is a single floating point value, which is the steering angle (model.py lines 79-96).

The model uses ReLU activations introduce nonlinearity (model.py lines 80, 84, 87, 91, 93, 95), and the data is normalized in the model using a Keras lambda layer (model.py line 76).

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer after the first convolution layer in order to reduce overfitting (model.py line 82). The layer was introduced at this position to induce the model to generalize better from noisy raw visual input.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 100). It was found that after about 3 epochs, validation error did not continue to decrease, so an "early stopping" regime of 3 epochs was used to prevent overfitting (model.py line 65 & line 100). The model was tested by running it through the simulator and ensuring that the vehicle would stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 99).  I used batch sizes of 128, which are large enough to speed up training but small enough to fit in memory (model.py line 64 & line 100).

#### 4. Appropriate training data

I used the Udacity-provided raw data. Training data was chosen to keep the vehicle driving on the road. Training data consisted of the center camera image and raw steering angle, the left camera image and raw steering angle plus a correction factor, the right camera image and raw steering angle minus the correction factor, and all of these with the images flipped across the vertical axis and the negative of the corrected steering angle values.

The left and right camera images and their flipped counterparts were used in place of recovery driving from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as follows:

1. Use center camera images and raw steering angles (and their flips) to train a simple model that is able to drive for a short distance.
2. Use this model to calibrate the correction factor for steering angles corresponding to the left and right camera images (and their flips).
3. With the correction factor calibrated, use the full set of camera images and steering angles to train a more complex model that drives better.

**Step 1: Train a simple model on center camera data.**

The LeNet architecture works pretty well for a variety of image tasks. So my first step was to implement the LeNet architecture and see how well it works on cropped center camera data, augmented with flipped data. Two modifications made to the model were to change the input size and replace the final softmax with a single dense connection. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Since predicting the steering angle is a regression task, I used mean squared error as the loss function. To measure how well the model was performing at each epoch, I used the `mae` mean absolute error metric. This is a reasonable metric to use because I want to know "how far off from the true steering angle" the model's predictions are, and both overpredicting and underpredicting are bad.

I trained the model for 7 epochs after which point validation error stopped decreasing. I saw that both training and validation error were fairly low, so I loaded the model in the simulator to see how well the car drove. With some tuning, I was able to get the car past the bridge before going off track.

**Step 2: Calibrate the steering angle correction factor for the left and right camera images.**

Now that I had a somewhat reliable model trained, I used it to calibrate the steering angle correction factor (see part 3 of this section for the definition of this value). I did so by repeatedly setting a correction factor value, training a LetNet model with it, and seeing how well it drove. Too much wobble meant that either the model was overreacting or underreacting to curves. Once I narrowed in on a value that had very low wobble (I found 0.3 to work well), I set it as the final correction factor value (model.py line 10).

**Step 3: Retrain an improved model with the full suite of data.**

With the correction factor calibrated, I set about improving the model. Noting that the car drove with a definite wobble, I first tried to add another convolutional layer, being careful to increase the layer's depth (I used a depth of 32) to capture the increased volume of information flowing through. Each successive layer in a convolutional neural net seems to capture increasingly complex shapes, so the thought was that if 2 layers was insufficient to eliminate wobbling, perhaps a third layer would combine shorter curve segments into longer curves and reduce the wobble.

After training a model with "early stopping," I found that while the car could successfully do a lap around the track, there were places where the car got uncomfortably close to the lane edges of the lane. Perhaps the decision-making end of the model could be improved? I noticed that the NVIDIA architecture uses not 2 but 3 hidden layers, of sizes 1000, 100, and 10, so I implemented this change and retrained the model. The performance was much better. To get rid of the last bit of wobble, I added a dropout layer after the first convolutional layer, to try and smooth out visual noise. This worked.

At the end of all this, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 79-96) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        															| 
|:---------------------:|:-------------------------------------------------------------------------------------:| 
| Input         		| 160x320x3 image 																		|
| Cropping         		| outputs 90x320x3 image 																| 
| Normalization    		| 																						| 
| Convolution 5x5     	| 1x1 stride, valid padding, 6 feature maps, outputs 86x316x6 							|
| ReLU					|																						|
| Max pooling	      	| 2x2 patch, 2x2 stride, outputs 43x158x6												|
| Dropout				| 25% chance of unit being zero'ing														|
| Convolution 5x5	    | 1x1 stride, valid padding, 16 feature maps, outputs 39x154x16							|
| ReLU					|																						|
| Max pooling	      	| 2x2 patch, 2x2 stride, outputs 19x77x16												|
| Convolution 5x5     	| 1x1 stride, valid padding, 32 feature maps, outputs 15x73x32 							|
| ReLU					|																						|
| Max pooling	      	| 2x2 patch, 2x2 stride, outputs 7x36x32												|
| Flatten				| outputs 8064  																		|
| Fully connected		| outputs 1000																			|
| ReLU					|																						|
| Fully connected		| outputs 100																			|
| ReLU					|																						|
| Fully connected		| outputs 10																			|
| ReLU					|																						|
| Fully connected		| outputs 1																				|


#### 3. Creation of the Training Set & Training Process

The Udacity-provided raw data appears to consist of a half-dozen laps around the track in a clockwise direction using only center lane driving. It contains 8,036 rows of data; i.e. 8,038 x 3 = 24,108 images from the center, left, and right cameras together.

Here is an example image of center lane driving from the center camera:

![alt text][image1]

Since the car was only driving in a clockwise direction, there was a bias toward positive steering angles:

![alt_text][image2]

I corrected this bias by flipping the camera images across the vertical axis (and taking the negative of steering angles). Here is an example flipped center camera image:

![alt_text][image3]

Finally, since the part of the image above the horizon contains a lot of information not relevant for making predictions, and the hood of the car is constantly present across images, I cropped out these parts of the image so as not to distract the model during training. It looked like the top 50 pixel rows and bottom 20 pixel rows could be eliminated without affecting the view of the road (model.py line 68 & line 70). Cropping was done within the Keras model (model.py line 75) so that at test time, the images are handled in the same fashion. Here is an example of cropped center camera images:

![alt_text][image4]

Prior to feeding data into the model, it was normalized (model.py line 76).

To determine whether the model was over- or underfitting, on each training epoch, 20% of the data was held back in a validation set (model.py line 100). On the LeNet architecture, the ideal number of epochs was about 7; after this point, validation error stopped dropping.

Once I had a fairly good simple model trained, I wanted to make use of the left and right camera images, since doing so would expand the range of steering angles and impetuses that the model is be aware of. I did so by adding a correction factor onto the raw steering angle for the left camera image and subtracting the same correction factor from the raw steering angle for the right camera image. Then using the same model architecture, I calibrated the correction factor by repeatedly setting the correction factor, training a model, and running the model in the simulator until wobbling was minimized.

Here is an example of left, center, and right camera images:

![alt_text][image5]

And their flips:

![alt_text][image6]

Here are the same images after cropping:

![alt_text][image7]

And their flips:

![alt_text][image8]

At the end of this process I had 24,108 x 2 = 48,216 data points. While refining the model architecture, I again on each training epoch held back 20% of the data in a validation set. For the final model, the ideal number of epochs was about 3, after which point validation error did not continue dropping. I used an Adam optimizer so that manually training the learning rate wasn't necessary.