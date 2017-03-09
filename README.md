
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

This project contains the following files:
- CarND-Vehicle-Detection.ipynb which contain the code
- svc.p which is the trained support vector machine
- X_scaler.p which is the fitted standardization object
- /output_images for documentation
- /test_images for pipeline testing
- project_video.mp4 as input
- vehicle_detection.mp4 as output

[//]: # (Image References)
[image1]: ./output_images/vehicle_images.png
[image2]: ./output_images/non_vehicle_images.png
[image3]: ./output_images/histogram_rbg_hsv.png
[image4]: ./output_images/hog_features.png
[image5]: ./output_images/single_hog_features.png
[image6]: ./output_images/raw_detection.png
[image7]: ./output_images/raw_detection1.png
[image8]: ./output_images/heat_thresholding.png

[image9]: ./output_images/296draw_heat.png
[image10]: ./output_images/297draw_heat.png
[image11]: ./output_images/298draw_heat.png
[image12]: ./output_images/299draw_heat.png
[image13]: ./output_images/300draw_heat.png
[image14]: ./output_images/301draw_heat.png
[image15]: ./output_images/296heatmap.png
[image16]: ./output_images/297heatmap.png
[image17]: ./output_images/298heatmap.png
[image18]: ./output_images/299heatmap.png
[image19]: ./output_images/300heatmap.png
[image20]: ./output_images/301heatmap.png
[image21]: ./output_images/296labels.png
[image22]: ./output_images/297labels.png
[image23]: ./output_images/298labels.png
[image24]: ./output_images/299labels.png
[image25]: ./output_images/300labels.png
[image26]: ./output_images/301labels.png


[video1]: ./project_video.mp4


---
### Explore Dataset
For this project around 17000 vehicle and non-vehicle images of the size 64x64x3 were given. We use theses images to train my support vector machine. Since classical machine learning is applied here, features must be selected manually. In the following sections different features are discussed.

![alt text][image1]
![alt text][image2]

### Spatial Binning
A simple approach is simply to use the whole image, resize it and collapse it into a single dimension vector, which has proven to be sufficient as input for neural nets with fully connected layers on the MNIST dataset. The function `bin_spatial()` (code cell 'Feature extraction functions' line 4) was tested as vector, but was not included in the final pipeline. It didn't significantly improve the classification accuracy and used a good portion of the computation time.

### Histogram of Color
The histogram of the color channels can be a good indicator to separate a vehicle from a non-vehicle. Different color spaces can be tested, to see which is suitable for a classification task. We use the `color_hist()` function (code cell 'Feature extraction functions' line 11) to create histograms. The image below shows the histogram of the three color channels in RGB and YCrCb color spaces. There should be a small differences within classes and a big differences across classes for optimal classification. We can see that RGB gives good separation but all channels are almost the same and therefore redundant. In contrast the YCrCb color space has a more diverse distribution.

![alt text][image3]

### Histogram of Oriented Gradients (HOG)

HOG features are great to capture the shape of an object in an image because it is to some extend invariant to perspective. In the `get_hog_features()` function (code cell 'Feature extraction functions' line 25), we use `skimage.feature.hog()` to calculate the histogram of gradients. This function does some normalization by itself across blocks which proves to be more stable and has also a built in visualization. We explored different color spaces and different parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). As we can see below, again, RGB and YCrCb are compared. The parameters for the show HOG features are `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

![alt text][image4]


#### Final choice of HOG parameters
Since the feature vectors can become very long, feature extraction becomes very slow. We therefore chose `orientations=6`, `pixels_per_cell=(16, 16)` without suffering a significant accuracy drop. As recommended in the documentation a normalization over blocks is recommended, therefore `cells_per_block=(2, 2)` was kept. Other color spaces were also tested like HSV and HLS, the highest accuracy was achieved by the YCrCb color space.

![alt text][image5]

### Support Vector Machine
For the classification task a linear support vector machine is chosen. We use the `sklearn.svm.LinearSVC()` implementation.

Before the actual training, some preprocessing is done (code cell 'Preprocess data'). First, the data is shuffled and split into 80% training data and 20% test data. Then the features are extracted with `extract_features()` for the whole data set.
The final feature vector with the length of 744 consists of:
- 3 histograms of color (YCrCb)
- 3 histograms of oriented gradients (YCrCb)

We then use `sklearn.preprocessing.StandardScaler()` to standardize our feature vector, which means zero mean an unit variance. This step is important to give each feature roughly the same impact and prevents single features to dominate the whole feature vector. The `StandardScaler()` should only be fitted with the training data and and applied on the training data and test data.

Finally, we can train the SVM and we gain a test accuracy of 98.5%. The classifier and the scaler are saved to pickle files (`svc.p` and `X_scaler.p`) for reuse.


### Sliding Window Search
To use the classifier on the whole image, we use a sliding window technique (code cell 'Vehicle detection'). First, we restrict the area of search to `y=[400:656]` since we know cars have to be on the ground. For further optimization the HOG features are calculated once for the whole area, so we can reuse parts of it, when we slide the window. Features are extracted as discussed previous sections with every window slide. We apply different window sizes to check for cars of different distance and repeat the feature extraction.

The sliding window search has two parameters we can tune. `cells_per_steps` define how far we slide our window and therefore determine how much the windows overlap. The second parameter is the scale of the window and how many different sizes we want to search in the image. Both parameters affect computation time and accuracy. For safely find all cars I chose to step two cells per step and scale the window from 1 to 2 with 0.2 steps.

In the image below, we can see all positive detections of different scales:

![alt text][image6]
![alt text][image7]


As we can see there are multiple detections for one car and some false positives. By using a heatmap and thresholding we can get an image like below. In the following section this step will be decribed in more detail.
![alt text][image8]

### Heatmaps

The positions of positive detections in each frame of the video are recorded. From the positive detections a heatmap is created. Detections over multiple frames at the same position give high values in the heatmap, whereas detections in a single frames will be thresholded. Also a 'cool down' applied, which means that the values in the heatmap will be reduced if they aren't confirmed in the following frames. `scipy.ndimage.measurements.label()` is used to identify individual blobs in the heatmap. Each blob should be a vehicle and bounding boxes are constructed to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Bounding boxes       |  Heatmap            | Lables      
:-------------------:|:-------------------:|:-------------------:
![alt text][image9]  |![alt text][image15] |![alt text][image21]
![alt text][image10] |![alt text][image16] |![alt text][image22]
![alt text][image11] |![alt text][image17] |![alt text][image23]
![alt text][image12] |![alt text][image18] |![alt text][image24]
![alt text][image13] |![alt text][image19] |![alt text][image25]
![alt text][image14] |![alt text][image20] |![alt text][image26]


Here's a [link to my video result](./vehicle_detection.mp4)

### Summary: Pipeline

Now the pipeline is complete. Here are the steps taken:
- Convert to YCrCb color space
- Create HOG over area of search
- Slide windows with different scales
- Get color histograms for window
- Get HOG for window
- Join features to feature vector
- Standardization
- SVM classification
- Update heatmaps and threshold+cooldown
- Draw final bounding box around detected vehicle

---

### Discussion

We used classical machine learning to find vehicles. An advantage of this approach that it is quite fast to train. But the classifier detects many false positives which have to be filtered out. Since the car is driving fast, false detections don't stay at one place. If we would move slower and false detections occur at the same location in the frame, we might get false positives with high heat.
Current state of the are techniques use deep learning methods to detect cars. These methods are more robust and are able to detect cars in context with the road.

The current prediction step of this algorithm is quite slow and reaches only 2 fps. It might be possible to further reduce redundant features to keep the support vector machine small and reduce the time for feature extraction.
