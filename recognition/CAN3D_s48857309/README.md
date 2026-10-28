# CAN3D: Anatomical Countouring in 3D for Radiotherapy planning
The CAN3D model is a model designed for semantic image segmentation (segmenting 
an image on a pixel-by-pixel level), for 3D image volumes. The model uses dilated 
convolutions to expand its receptive field, so that it can use both local and global context
to make predictions. Compared to other models, it has lower memory and computational 
costs, which is relevent to applications which are deployed in memory and gpu limited
clinical imaging systems. We are using this model to perform real-time segmentation
of 3D MRI images of the prostate for prostate radiotherapy planning. We are interested in whether
the 3D context provides significant benefit over baseline 2D models, ideally by using 3D 
context to avoid hazardous failures such as failure in segmenting accurately top and bottom slices 
(which are likely under-fitted due to the fact that these slices account for only a small percentage of the 2D slice data),
and avoiding leaking across the thin interface of the prostate into nearby organs.



## Feasiblity Report
### User Need, Scope & Acceptance Criteria
The intended users of our model are imaging specialists and radiologists who
would like assistance in segmentation for real-time prostate radiotherapy planning, and in
order to save clinical countouring time. The core problem for this model is that common metrics such as the dice coefficient
can often achieve a high accuracy while failing at the boundary of the organ, where boundary leakage can cause severe side effects for patients, such as radiation induced inflammation of the bladder or rectal lining.
Additionally, models can often achieve good average accuracy while having catestrophically low accuracy on the base and apex slices (especially for 2D models), because central slices often contain easily recognisable broad regions.

Core problems that need to be solved are:
* Achieving good results on boundaries.
* Achieving uniformly good results on all slice planes (including base and apex slices).
* Near real-time efficiency on inference for GPU limitted systems.

The purpose of the prototype is to see initial results on partially trained outputs to 
see if further training will yields results worth the additional computaitonal expense, to see that
training is at least stable in early epochs, and whether the proposed architecture is likely to perform better
than simpler baselines in terms of accuracy and inference time.

I will use the following acceptance criteria:
* Segmentation of the prototype model visually shows reasonable gains on base and apex slices compared to the 2D baseline.
* Runtime of inference occurs in under 2 seconds on a CPU.
* Peak RAM usage on inference is less than 2 GB.
* Achieving a dice coefficient of over 0.7 overall for the unseen test dataset.
* Initial graphs of training epochs show steady progress, and training and test loss converge with minimal overfitting.


### Model Choice & Course Concepts
The model I chose was stated as having HARD difficulty.

The model I have chosen is to use a CAN3D model, which works by applying 
three-dimensional dilated convolutions to rapidly expand the receptive field of 
each pixel/voxel to enable the aggregation of global context clues in the images 
to assist in segmentation, while retaining low memory requirements and inference times.

The use of full 3D images is intended to improve upon 2D slices by assisting the 
model by, in particular, identifying the boundaries and lower and upper extremal slices
using the broader three-dimensional context to better guide segmentation. This is
 likely to work since a lot of information is lost when only viewing one 2D slice at a time.

The model architecture consistents of an initial 3x3x3 convolution with 3 channels, a skip connection,
two downsampling kernels with output channels 6 and 36 respectively, a Context Aggregation Module (CAM),
which consistents of 3 dilated 3d convolution layers of increasing dilation factor (2, 4, 8, respectively) and 
a layer with a dilation of 1 increasing the number of channels to 48. This is then upscaled twice using transposed convolutions
with a stride and kernel size of 2, and concatonated to the channels of the skip connection. Each of these above convolutions contain 
what the authors refer to as "Adaptive Instance Normalisation" (which is a layer that is equal to a weighted sum of the instance norm and the identity map of the inputs) and leaky relu layers between them. Then a single 3d convolution layer outputs the logit predictions, which are then passed through softmax for final estimate probabilities.

The use of LeakyReLU is justified by the fact that it ensures that there are always gradients, and you avoid the dead-neuron problem for simple ReLU.
Adaptive Instance normalisation is used to improve training stability, improve convergence, and limit internal covariance shift (which occurs when the distribution of inputs changes as the model learns, which can cause training instability). 

![Diagram of model architecture](model_architecture.jpg)
(W. Dai, 2021)

The model also makes use of a unique loss function. In particular, the loss function is a combination of Dice Coefficient Loss and Focal Loss (which in the literature was originally used for dense object detection), 
which tends to perform better at segmenting boundaries than Dice Loss alone (combined, they are called Dice Squared Focal Loss, DSFL). The total loss is the weighted sum of DSFL losses of each class. The focal loss is 
similar to log loss for logistic regression, except that the loss is weighted more heavily for samples which have high uncertainty. The effect of this is to hopefully improve the predictions at difficult boundary cases.

The dataset contains 211 images in total, with many images belonging to the same patients. Since we are using a fixed model, and we don't have hyperparameters necessary to optimise, then we only need a test set (validation set is not necessary).
So, we perform an 80-20 split of training to test data. This split percentage is often a good rule of thumb, especially for reasonably small data sets such as this. When doing the test split, we make sure not to leak data between the sets by assigning samples so that samples belonging to the same patient are put into only one of the sets. If we were to mix these samples together without considering the patient level splitting, this would bias the validation accuracy estimate, since scans from the same patient are likely to be similar to previous scans, and so the model can memorise this data point rather than learning the real patterns.

### Preliminary Feasibility Evidence
The dataset contains accurately segmented MRI volumes, identifying 5 segmentation classes (Body, Bones, Bladder, Rectum, Prostate), and 0 as the background. Each volume has a dimension of exactly 256x256x128. All data contains roughly centered scans, with similar positioning and scale. There are also 211 samples, with between 1 and 7 weeks worth of scans per patient. The size of samples is too much to fit large batches into VRAM. If we use a 80-20 split for training and testing, this means that we will have 42 test cases. This means that test accuracy statistics (for example) would have a worst-case standard error of about 0.04. 

After some testing on an initial implementation, I found that on epochs of 10 samples per epoch, each epoch took around 20 seconds each. Running a small smoke test on the model, only running it for 10 epochs on 10 samples per epoch, I achieved the following images:

![True Image](true_image.png)
![True Labelling](true_labelling.png)
![Smoke Test Prediction](predicted_labelling.png)

We can see clearly that the small test has results that are extremely inaccurate at this stage, but that it is not experience major instability or model collapse. By fine tuning the training parameters (such as class weights in the loss), and training for longer, we may find better results. However, at this state, there is a lot of progress to make.


### Risks, Budget & Fallback
The main risks are assosciated with negative health outcomes due to innacurate segmentation.

Some key technical risks include:
* Innacurate segmentation can lead to serious health consequences, so high accuracy is required.
* Only small batch sizes being able to fit into memory, which reduces training speed. This may make training better models difficult.
* Low amount of data, making generalisation more difficult. Augmentation may be possible, but limited.

Computational requirements:
* From tests, it took 100 seconds to train 5 epochs on 10 samples per epoch. This means that to train a 50 epoch model on all 211 samples, it would take roughly 5 hours.
* To simply store a single image for testing, it takes 6MB, so storing the whole database takes 1.2GB for the data.

My next planned experiment is to test with the full dataset, for a larger number of epochs, and using data augmentation with random translations. I also want to experiment with different more basic models.

A possible fallback if my planned experiment does not work, is to use the baseline model instead, assming that it has reasonable performance. 


## Dependencies and Reproducibility
Python: 3.11.17,
torch: 2.14.1 (cuda enabled),
numpy: 2.4.6,
matplotlib: 3.11.2,
monai: 1.6.1,
nibabel: 5.4.2,
scikit-learn: 1.9.1,
tqdm: 4.70.1

Splits are generated with random_state=42. Training the model should take on the order of 2 hours on a GPU. 
The model can be trained by calling `train.py`, and indicating the directory of the dataset in `dataset.py`.
All final trianing parameters and model parameters and contained in `train.py` and `modules.py`.

## Examples and Plots
The final model gave the following output predictions.

![3D prediction example](prediction_example_3d.png)

In this diagram above, yellow indicates the prostate. The left-most image shows a particular slice of an MRI taken from a 3D volume. The middle shows the expert-labelled segmentation, and the right shows the CAN3D model segmentation. We can see that the models prediction has a few spots that are incorrectly classified as background which contribute to some of the noise of the prediction. We see that in this case, the prediction for the bone class is actually more realistic than the expert labelling, since we see in our model's prediction, a continuous bone region, while on the expert-classified segmentation, the upper left corner shows some artifacts.

While training the model, we saw the validation loss curve below:

![Validation curves](validation_curves.png)

This plots the DSF loss on the left plot, and the Dice Coefficient for the prostate class on the right plot.

We see that near the beginning, the loss was near zero, then around epoch 10, there was a massive increase, followed by a relatively quick plateau. The loss curve shows gradual decline, while the validation score seems to peak at around epoch 60, and declines negligibly after that.

The final testing accuracy of the CAN3D and CAN2D model are displayed below for all classes. The prostate class is the most relevant to our investigation. Both models were trained on the same dataset using the same loss function, and compared on the same splits.

| Model | Statistic | Background - 0 | Body - 1 | Bones - 2 | Bladder - 3 | Rectum - 4 | Prostate  - 5 (Target) |
| ----- | --------- | ---------- | ---- | ----- | ------- | ------ | -------- |
| CAN3D | Dice Coefficient | 0.9787 | 0.9668 | 0.8577 | 0.8227 | 0.7785 | 0.7792 |
| | Intersection/Union | 0.9787| 0.9359 | 0.7515 | 0.7084 | 0.6484 | 0.6506 |
| CAN2D | Dice Coefficient | 0.9330 | 0.8401 | 0.5464 | 0.3323 | 0.2536 | 0.3205 |
| |Intersection/Union | 0.8748 | 0.7265 | 0.3847 | 0.2121 | 0.1537 | 0.2091 |


We can see that the 3D model massively outperforms the CAN2D model on this problem. See also that CAN3D reached a Dice Coefficient greater than 0.7 across the board.

Below I show a sequences of slices from the same volume, shows first the apex slice, middle slices, and base slice, as predicted by the CAN3D model. The right-most image shows false positives for the prostate class in red, and false negatives in blue.

![Difference Image 1](difference_overlay_1.png)
![Difference Image 2](difference_overlay_2.png)
![Difference Image 3](difference_overlay_3.png)
![Difference Image 4](difference_overlay_4.png)
![Difference Image 5](difference_overlay_5.png)

We see that all slices except for the base slice (the last image) show very good accuracy, where for the base slice, it makes many false positive predictions for the prostate.

## Pre-Processing
The following pre-processing steps were taken during the training process:
- Cropping every image to the same size 256x256x128. (Most images were unaltered)
- Each image undergoes standardisation independently, so each image has the same range of values.
- Augmentation with random elastic deformation using the Monai library, and random flipping along the body's natural bilateral axis. This was done to improve generalisation performance.

During testing and model validation and evaluation, only cropping and standardisation was applied.

The data was split into 3 sets, training, validation, and testing, with 70-10-20 percentages respectively. 
Scikit-learn's GroupShuffleSplit was used to split on a patient-level, to avoid data-leakage.
A 20% split was chosen for testing to get a reasonably accurate measure. Since the dataset has 211 samples, this means that roughly 40 samples are used for final model evaluation, which should give a reasonable estimate of the accuracy of the model. The final splits used end up with 122-25-64 samples respectively.
The 20% test split was not used while training and tuning models, so as to not bias or overfit to it, and instead the 10% validation split was used consistently through the tuning and choice of the final model.

## Open Research Dilemma
The core investigation of this project was to determine whether incorporating true 3D volumetric continuity would reduce or eliminate hazardous boundary and slice-transition errors compared to 2D models. I make a comparison between the CAN2D model (F. Yu, 2015), and the CAN3D model (W. Dai, 2021), which both use dilated convolutions to aggregate contextual information, but where the CAN3D model uses full 3D cnvolutions, while CAN2D operates slice-wise.

### Baseline Comparison
### Computational Efficiency
### Errors
### Recommendations

## Journal and Personal Reflection
Before getting the final CAN3D model, I attempted training using several different training methods, loss functions, and slight changes to the architecture. I also made several big mistakes while implementing this model and training that I had to correct before getting good results. I also made adaptations to improve performance for the final model. 

1) The first model I trained used the CAN3D architecture suggested by the paper, with TransposedConvolutions for upsampling, and concatenating the skip connection. This model performed extremely poorly, after training for 20 minutes, it only game given a constant prediction for the 'body' class (which is the most common class).
2) I found a mistake in that implementation, where I found that the Adaptive Instance Norm layers were not the AdaIn layers seen in models like StyleGAN, but were defined as a weighted sum of an Instance Norm and an Identity. 
3) After changing the upsampling channel count to 3, and summing rather than concatenating the skip connection (because I felt that reducing the number of parameters would reduce the model complexity, so would improve training stability). This model was the one provided for the example run in the feasibility report. As seen, this produced pretty chaotic results.
4) I found a bug in the Dataset implementation which was producing blanks, fixing this made a big difference for obvious reasons.
5) I then started implementing the baseline, since I was getting errors in the 3D model I could not figure out. In retrospect, I should have done this first. 
6) After adding the correct implementation of the identity initialisation for the CAM module, I got my first decent model for CAN2D. I still had a major mistake in this implementation.
7) I finally found the big mistake I was making, where my implementation for Focal loss term in the DSF loss was incorrect. I was missing a major '* y_true' term.
8) After fixing this issue, I tried training with varying hyperparameters, but I still couldn't get a good result. I added an additional layer to the CAM module to increase the perceptive field of the model, and this produced marginally better results, but still not very good. 
9) To get a basic working CAN3D model, I tried simplifying the problem by downsampling via MaxPool, and doing a simple nearest neighbour upsample, and I removed the TransposedConvolution layers. This produced the first reasonable results of any model I had attempted to this point.
10) After adding image augmentation to training, I re-implemented the old kernel-stride-based downsampling, but used nearest-neighbour upsample layers with convolutions rather than TransposedConvolution layers. This massively improved training stability compared to previous models.
11) Then, to produce the final model, I played with the number of channels of the upsampling layers, and increased the weighting of rarer classes in the loss function (specifically, Rectum and Prostate class weights were massively increased). I kept increasing the weights of prostate until I got the final model with Dice Coefficient > 0.7 across the board. I noticed that as I increased the weights for the prostate, as expected, the background and body dice-coefficients got worse (which is a trade-off induced by the loss function weights).

I tested many other variations, but these were the main milestones I committed.

## References
W. Dai, S. Chandra, et al. “CAN3D: Fast 3D medical image segmentation via compact context aggregation”, 2021, https://doi.org/10.1016/j.media.2022.102562.
Olaf Ronneberger, et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015, https://doi.org/10.48550/arXiv.1505.04597
F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions”, 2015, https://doi.org/10.48550/arXiv.1511.07122 

## Artificial Intelligence Usage Disclosure
I have not used AI tools to help program or give ideas for any part of this project.