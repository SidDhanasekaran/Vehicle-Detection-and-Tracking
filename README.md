# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Introduction

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.

## Detailed description

The code is contained in the IPython notebook `p5_vehicle_detection_latest.ipynb`. 

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/spatial.jpg
[image3]: ./output_images/color_histogram.jpg
[image4]: ./output_images/HOG_Visualization.jpg
[image5]: ./output_images/Normalize.jpg
[image6]: ./output_images/windows.jpg
[image7]: ./output_images/detect.jpg
[image8]: ./output_images/heatmap.jpg
[image9]: ./output_images/boundingboxes.jpg

## Import the input data

(The code is contained in `cell #2`)

I used the images provided by Udacity as my input data. I spilt the various folders to create the training and testing sets as follows: 

**Training set**

- cars: 
        -GTI_far
        - GTI_left
        - GTI_right
        - GTI_MiddleClose 
- non-cars: 
        - Extra

**Test set**

-  cars: 
        - KITTI_extracted 
-  non-cars: 
        - GTI

I printed out some basic information of the data set such as the number of the image in each class and  image size. Below are some data from my input images: 

**Training Data**
Car images:     5339
Non-car images: 5068
Image size:     (64, 64, 3)

**Test Data**
Car images:     2826
Non-car images: 3900
Image size:     (64, 64, 3)

![alt text][image1]


## Define Features

Three types of features are used: 

- Spatial feature
- Color histogram features 
- HOG features.

## Spatial Feature

(The code is contained in `cell #3`)

The images in the training data set are of the jpeg format, with float data values range from 0-1. The test images are of the png format, with int data values range from 0-255. To be consistent with the images type in the later process. I first convert the training image data type to int type with value from 0 to 255.

The spatial feature uses the raw pixel values of the images and flattens them into a vector.  I performed spatial binning on an image by resizing the image to the lower resolution. To reduce the number of features, only the saturation channel in the HLS color space is used, based on the assumption that the saturation channel would be a good representation of the image, because the cars are more likely to have a more prominent appearance. Here is an example of an image in Saturation Channel and the value of the Spatial features.

![alt text][image2]

## Color Histogram Features

(The code is contained in `cell #4`)

Color Histogram feature is more robust to the various appearances of vehicles.  The Color Histogram removes the structural relation and allows for more flexibility to the variance of the image. Binning is performed to the histogram of each channel. Both the RGB and HLS channels are used. Here is an example of the color histogram feature in GRB and HLS color space.

![alt text][image3]

## Histogram of Oriented Gradients (HOG)

(The code is contained in `cell #5`)

The Histogram of Gradient Orientation (HOG) is also used to capture the signature for a shape and allows variation. The HOG is performed on the gray scale image. Here is an example of the HOG feature.

![alt text][image4]

## Extract Features

(The code is contained in `cell #6`)

This step creates a pipeline to extract features from the dataset. The feature extraction parameters need to balance the performance and running time. After trial and error, I found the performance doesn't increase much after 1000 features. To keep algorithm run in real times, I keep the number of features around 1000. The feature extraction parameters are as follows:

**Spatial feature parameters:**

* spatial = 8 
* channels:  HLS and RGB
* number of feautures: 384

**Color histogram feature parameters:**

* hist_bins = 12 
* channels: HLS and RGB
* number of feautures: 72

**HOG feature parameters:**

* orient = 8
* pix_per_cell = 12
* cell_per_block = 2
* channels: Grey scale
* number of feautures: 512

**Total number of feature:** 968

## Feature Normalization

(The code contained in `cell #7`)

The `StandardScaler()` function is used, which removes the mean and scales the features to unit variance. The StandardScaler is applied to the training and testing set. 

Here is an example of the raw and normalized feature.

![alt text][image5]

## Create the Training, Testing, and Validation set

The image in the training set is randomly shuffled. The image in the testing set is divided equally into testing set and validation set.

**The number images:**
Training set  :  10407
Validation set:  3363
Testing set   :  3363

## Create the Classifier



(The code is contained in `cell #9`)

Random forest algorithm was chosen because it has a good balance of performance and speed. The algorithm uses the ensemble of decision trees to give a more robust performance. The tuning parameters including  max_features, max_depth,  min_samples_leaf. `auroc`is used as the performance metric to measure the robustness of the algorithm. 

A grid search is conducted to optimize the parameters. smaller max_features have better performance because the classifier tends to find more general rules. The final set of parameters are as follows

n_estimators = 100
max_features = 2
min_samples_leaf = 4
max_depth = 25

The performance of the classifier on training testing, and validation set is shown as follows:

Training auroc    = 1.0
Training accuracy = 0.9998
Testing auroc    = 0.9709
Testing accuracy = 0.8046
Validation auroc    = 0.9682
Validation accuracy = 0.8011


## Vehicle Detection

(The code is contained in `cell #10`)

Using the classifier on sliding windows to detect whether an image contain cars.Sliding windows are used to crop small images for vehicle classification.To minimize the number of searches, the search area is retained to the area where vehicles are likely to appear.The minimum and maximum size of the window are decided. The intermediate sizes are chosen by interpolation. 

Here is an example of search windows with different size.

![alt text][image6]


## Extract Features form Windows

(The code is contained in `cell #12`)

The pixels in each window are cropped and rescaled to 64x64x3, which is the same as the training image. Then the classifiier determines if the window has a car or a non-car in it. Here is an example shows window of the detected vehicle for all the test images:

![alt text][image7]

The classifier misses the car with darker color. Proablby because the color of the car is not very prominent. But, overall the classifier does a good job in finding cars images

## Duplicates Removal

(The code is contained in `cell #15`)

To eliminate the duplicate boxes, a heatmap is built from combining the windows which have car detected. Then a threshold is added to filter out the False Positives. Since False Positives are not consistent. It will be more likely to appear and disappear. After the heatmap is thresholded. Use 'label' to find all the disconnected areas. Here is an example shows the heatmap box and labeled areas.

![alt text][image8]


## Find Bounding boxes of labels

(The code is contained in `cell #16`)

A bounding box is estimated by drawing a rectangle around the labeled area. Here is an example shows window of the bounding box:

![alt text][image9]
 
 
## Vehicle Tracking

(The code is contained in `cell #17`)

I have used a `car` object to track the detected cars, which contrains 4 attributes, `average_centroid`, `width`, `heigh`, `detected`. 

The the tracking process is as follows:

- In each frame, a new heat map `heatmap_new` is created for the window that contains car images. 
- The the global variable  `heatmap` is updated using the moving average method.
- It is a weighted average of the previous average and the new value.The advantage of this method is that it doesn't need to store all the previous values, and only keeps the value of the previous average. The old value decreases exponentially and fades out. The `heatmap_sure` thresholds the heatmap to show result with more certainty and creates `bounding_boxes`
- After finding the bounding boxes. I calculate the distance between the centroid of the bounding box to the centroid of previously detected cars to see if there is a nearby car object.
- If the distance is within a threshold. It updates the previous car object, with the new centroid, width, and height using the moving average method.
- If no car is found nearby, I create a new car object. Then I combine `new_cars` to `Detected_Cars` and loop through the previous `Detected_Cars`.  
- If the detected value is greater than the threshold, it is kept and if not discarded.
- Finally, I depreciate the detected values, so if a car is no longer detected the value decreases exponentially.



### Vehicle Tracking Pipeline

 The pipeline performs the vehicle detection and tracking on each frame.The results are visualized and overlaid on the original images:

- Detected windows: Blue boxes
- Heatmap: Green area
-  Bounding boxes of cars: Red boxes


## Discussion

Some of the Challenges or improvements needed are below: 

- From the final video results, the classifiers still gets many False Positives, especially around the fences on the left side. It possible the fences have vertical lines which can be confusing to car images. By setting a threshold on heatmap, I was able to reduce many of the False Positive. 
- The moving average method used to update the position of the bounding box also introduce some delay, as we can see the bounding box is "lagged" behind the vehicle.


