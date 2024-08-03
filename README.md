## Variation Regions of Parameter Set Images

Parameter Set images are loaded from a specified root directory and coordinates are loaded from a vertex file in txt format. The loaded coordinates are processed to remove any potential outliers.

For each image, coordinate changes between the current and previous images are calculated. These changes are detected if the Euclidean distances exceed a certain threshold. 
Then, a segmentation mask is created for each image. In this mask, the original coordinates are marked in white and detected changes are highlighted.

Finally, change detection is performed on a specified number of samples, and these samples are visualized. For each sample, the original image, the segmentation mask, and the overlay of the mask on the image are saved. 
As a result, the detected changes are saved and visualized. This process consists of steps for loading images, processing coordinates, detecting changes, and visualizing results.

![img1](https://github.com/kursatkomurcu/fitspace_task/blob/main/images/Plot_21_Camera_Front.png)
![img2](https://github.com/kursatkomurcu/fitspace_task/blob/main/images/Plot_47_Camera_45.png)

## Plan for Training a Siamese Network for Change Detection

Our goal is to train a Siamese network that can detect changes between pairs of images. The network will learn to differentiate between image pairs that have changes and those that don't, using a contrastive loss function.

### Data Preparation

The images are stored in batches, each with unique IDs and camera types. We also have coordinates for key points which we'll use to generate segmentation masks.

We'll load images from the provided paths and preprocess them by converting them to grayscale and resizing them to 256x256 pixels. We'll create segmentation masks using the key points from the dataframe.

Generate segmentation masks for each image based on the key points. Align the masks using feature matching techniques like ORB to ensure consistency.

### Data Feeding Strategy

Each training instance will consist of a pair of images: one from the current time step and one from the previous time step. For each pair, we'll also generate corresponding segmentation masks.

Create binary labels indicating whether there are changes between the image pairs (1 for changes, 0 for no changes).

Split the data into training and validation sets using an 80/20 split to ensure the model can generalize well.

### Training Strategy
#### Hyperparameters
We'll use the contrastive loss function to train the network. This function encourages the network to produce similar feature vectors for similar images and different vectors for dissimilar images.

```
ContrastiveLoss = (1 - y) * 0.5 * (D_W)^2 + y * 0.5 * max(0, m - D_W)^2
```

where y is the label, D_W is the Euclidean distance between the feature vectors, and m is a margin parameter.

Use the Adam optimizer for training the network.



