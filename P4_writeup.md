
## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistort_output.JPG "Undistorted"
[image2]: ./writeup_images/dist_undist.JPG "Road Transformed"
[image3]: ./writeup_images/binary_combo_example.JPG "Binary Example"
[image4]: ./writeup_images/warped_straight_lines.JPG "Warp Example"
[image5]: ./writeup_images/color_fit_lines.JPG "Fit Visual"
[image6]: ./writeup_images/example_output.JPG "Output"
[video1]: ./laneVideo.mp4 "Video"

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./adv_lane_proj.ipynb" (from code block 1 to 3).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in function sobel_with_color_chnl ).  Here's an example of my output for this step.

![alt text][image3]

#### 3. perspective transform.

The code for my perspective transform includes a function called `perspec_matrix()` and `warped_img()`, which appears in in the 41st code cell of the IPython notebook.  The `perspec_matrix()` function takes as inputs an image and with the help of defined `src` and `dst` points it returns M and Minv.  I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 450      | 300, 0        | 
| 695, 450      | 950, 0        |
| 1090, 680     | 950, 719      |
| 230, 680      | 310, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
`warped_img()` function takes binary image as input and returns perspective(bird eye view) image of the same.

![alt text][image4]

#### 4. Identification of lane-line pixels and fit their positions with a polynomial.

Then I used Windowing approach to find lane line pixels. As windowing is computation intensive process so it is being used only once and in subsequent steps we have used polynomial offset window to identify lane line pixels because the frame rate is high so lane lines are not going to change abruptly.Then lane lines pixels are fitted with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 162 through 188 in my code in which appears in in the 41st code cell of the IPython notebook.

#### 6. Lane area identification.

I implemented this step class method `draw_lane_polynomials()` which inturns defines the pipeline of the project.It takes a normal RGB lane image. This method calls others functions internally.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Final video output.

Here's a [link to my video result](./laneVideo.mp4)

---

### Discussion

#### 1. problems / issues faced 

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

* Choosing only window based approach was taking much time for video processing,Skipping the windowing part for the frames(except first) resulted in faster processing.
* Other color channel can be xplored for better filtering.
* To remove lane area fluctuation I have used averaging (10 frames) approach,which resulted in smoother lane area drawn on image.
* Predictive approach (last lane lines co efficients) can be used to improve code performance on poorly visible lane lines.
