#Advanced Lane Finding Project - Computer Vision

This is advanced lane detection project where I have used Computer Vision - OpenCV techniques to identify lane boundaries and compute the estimate the radius of curvature given a frame of video of the road. Below are the goals and steps taken in this project:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./example_images/undistorted1.png "Undistorted1"
[image2]: ./example_images/undistorted2.png "Undistorted2"
[image3]: ./example_images/undistorted3.png "Undistorted3"
[image4]: ./example_images/undistorted4.png "Undistorted4"
[image5]: ./example_images/undistorted5.png "Undistorted5"
[image6]: ./example_images/undistorted6.png "Undistorted6"
[image7]: ./example_images/undistorted7.png "Undistorted7"
[image8]: ./example_images/undistorted8.png "Undistorted8"
[image9]: ./test_images/test1.jpg "Test1"
[image10]: ./example_images/threshold1.png "Threshold1"
[image11]: ./example_images/transform1.png "Transform1"
[image12]: ./example_images/transform2.png "Transform2"
[image13]: ./example_images/linefit1.png "Linefit1"
[image14]: ./example_images/linefit2.png "Linefit2"
[image15]: ./example_images/final1.png "Final1"
[image16]: ./example_images/final2.png "Final2"
[image17]: ./example_images/final3.png "Final3"
[image18]: ./example_images/final4.png "Final4"
[image19]: ./example_images/final5.png "Final5"
[image20]: ./example_images/final6.png "Final6"
[video1]: ./videos/proc_project_video.mp4 "Video"



To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/571/view)


Applying gray scale image to the original image using cvtColor() function of opencv.

<img src="examples/Step1.JPG" width="480" />

###CAMERA CALIBRATION AND DISTORTION CORRECTION

The transformation of a 3D object in the real world to a 2D image isn't perfect. We have to correct image distortion because it changes the apparent size, or shape of an object and more importantly it makes objects appear closer or farther away than they actually are.

A chess board can be used because its regular high contrast patterns makes it easy to detect and measure distortions as we know how and undistorted chessboard looks like. If we have multiple pictures of the same chessboard against a flat surface from the camera, we can get the from the difference between the apparent size and shape of the images compared to what it should theoretically be. We can create a transform that map the distorted points to undistorted points, and use this transform to undistort any image, which will be discussed more later.

Check [Camera Calibration](./camera_calibration.py) and [Camera Calibration Notebook](./camera_caliberation_with_chessboard.ipynb) for the implementation details. We save the parameters to a pickle file [Camera Calibration Pickle file](./camera_cal/cal_pickle.p), so we can use it later.

Here I have prepared "object points", comprising the (x, y, z) coordinates of the chessboard corners. It's assumed the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.Have updated the code to identified all the corners of the various chessboard images and then capturing the "mtx", "dist" . Once find corners is completed , introduced the `objp` hyper-parameter which is a replicated array of coordinates, and `objpoints` hyper-parameter will be appended with a copy of it every time .

`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Here `9 corners` in `x-axis` and `6 corners` in `y-axis` is used. However , since some of the chessboard images have separate dimentions , have used a range of +1 2 for x,y axis .

Then , have used `cv2.calibrateCamera()` function inducting the output of `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients. The distortion correction parameters is used to the test image using the `cv2.undistort()` function. 

![alt text][image1]
![alt text][image2]

The calibration result is saved as a `pickle` for future uses. [Camera Calibration Pickle file](./camera_cal/cal_pickle.p)


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

The comparison between the test images and their undistorted images are shown below. The differences are not apparent, but have few corrections.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

One more example, to describe how the distortion correction is applied to one of the test images:

![alt text][image9]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Here, a combination of color and gradient thresholds is used to generate a binary image (check steps under point `4` in the [IPython notebook](./pipeline.ipynb). Below are the thresholds :

 1. `Sobel` on `x` direction
 2. `Yellow` and `white` color detection from `RGB`
 3. `Yellow` and `white` color detection from `HSV`
 4. `Yellow` and `white` color detection from `HLS`
 5. `S channel` from `HLS`

![alt text][image10]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

####GETTING THE ???SKY VIEW??? WITH PERSPECTIVE_TRANSFORM

Given an undistorted image of the vehicle???s perspective vehicle_view we can warp this image to output an image of another perspective such as a bird???s eye view from the sky sky_view given a transformation matrix. We can derive transformation matrix warp_matrix by giving the pixel coordinates of points of the input image of one perspective source_points and the corresponding pixels coordinates of the output perspective destination_points using the function getPerspectiveTransform(). When we use this warp_matrix to output an image to another perspective from a given perspective using the warpPerspective() function. You can get the inverse_warp_matrix to get from ???sky_view??? to ???vehicle_view??? by switching the places of the source and destination points fed into the function getPerspectiveTransform().

The code for my perspective transform is located under the [Birdseye.py](./birdseye.py). The `PerspectiveTransform()` function uses pixel locations for both the original and new images and calculates the `M` and `Minv` matrices for perspective transform. Then, the `M` matrix is used by `cv2.warpPerspective` to convert the original image to its bird's view, while `Minv` is used to do transform in the opposite direction. Below is an example of the outpuyt:

![alt text][image11]

Another example of a transformed image adding threshold:

![alt text][image12]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Refer to the code for the line detection iin point `5` in the notebook [lane-detection](./lane_detection.py). A histogram is used to identify the locations of the maximum pixels along the `y-axis` and assume those are the locations of the points on the line. A `RejectOutlier` function is used to compare the `x-axis` value of all the points to their median value and reject the ones who are far away from the median. Last, a 2nd order polynomial is used to fit the lane lines such as:

![alt text][image13]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The calculation of the radius of curvature of the line is obtained from the lecture. The vehicle position is determined by calculating the average `x coordinate` of the bottom left and bottom right points of the lines and comparing it with the middle point of the `x-axis`(i.e., 640). The deviation is then converted from pixels to meters. If the deviation is postive, vehicle is to the right of the center. If the deviation is negative, vehicle is to the left of the center. An example is shown as below:

![alt text][image14]

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
The implementation of the whole pipeline is located under `6` in the [IPython notebook](./pipeline.ipynb). Below are the final images for all six test images:

![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
   
 1. How to use more deep learning techniques to predict the lanes better.
 2. How to tackle the harder challenge video
 3. How to make the vide convrsion to identify lanes more faster in realtime situations where it will be using live video from the cameras.
