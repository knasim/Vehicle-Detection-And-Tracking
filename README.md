## Vehicle Detection And Tracking

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/detection.png
[image9]: ./output_images/car_postions_heat_map.png

[video1]: ./project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in (main.py lines 18-27).  


#### 2. Explain how you settled on your final choice of HOG parameters.

Final choice for the chosen values of HOG parameters were mostly based on trial and error in order to arrive at a suitable value combination. For color_space (main.py line 18) I ended going with `YCrCb` because it had least false positives as opposed to `RGB`. Under varying light conditions RGB is not good.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a  `linear SVM` (main.py lines 70-73).  The classifier used is the `LinearSVC` from `scikit`.  To achieve standardization I applied a `StandardScaler`.  
Standardization of a dataset is a common requirement for many machine learning estimators otherwise they might behave badly if the individual feature do not more
or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially plenty of false rectangles were observed which required increasing the overlap.
This did help in reducing false rectangles but they were not eliminated. The number of false
rectangles reduced, however, they did not disappear completely.
As a consequence the HOG parameters and to be further manually tuned.

Here are some example images:
![alt text][image8]

---

### Video Implementation dddd
Here's a direct YouTube link [link to my video result](https://youtu.be/6HqLwyT6V50)

Here's a [link to my video result](./output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
There are functions in (function.py lines 361-392) that handle false positives by filtering using a heatmap as well as d etermining bounding boxes.

![alt text][image9]

---

### Discussion

A linear SVM/HOG approach as I learnt produces false positives and this is a challenge. Execution speeds frames per second (FPS) can be area of improvement.  
I also feel some aspects of the program that run sequentially could benefit from parallelization.  A CPU was used for this project.   False positives were not entirely eliminated
even after heatmap filtering, and this can be anther area for improvement.

I also took a peek at an advanced Deep net called YOLO (you only look once).  This could be a very exciting application for this project I think just see the comparision between
SVM + HOG vs. YOLO
