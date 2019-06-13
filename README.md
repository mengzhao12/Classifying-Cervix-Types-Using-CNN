# Classifying-Cervix-Types-Using-CNN

## Objective: Accurately Classify Cervix Types for Diagnosing Cervical Cancer Using Convolutional Neural Networks
 
This project uses convolutional neural networks(CNN) to classify cervix images into three types based on the location of transformation zones. While three CNN models are trained with original cervix images, the fourth CNN model is trained with processed images in which only the region of interest (ROI) is shown and the other irrelevant region becomes black in color. All the CNN models, in regardless of employing image segmentation, reached the similar accuracy of ~60% and sparse categorical cross entropy loss of ~0.9. This indicates all models’ performances are similar. We believe that the quality of the training data might be the bottleneck of improving the model’s performance.

## Selective Key Results

### architecture of CNN models:
------------

<p float="left">
  <img src="./figures/architecture_of_CNN.png" align="left" width = "100%">
</p>

### Loss and Accuracy of four CNN models:
------------

<p float="left">
  <img src="./figures/CNN1_model_accuray.png" align="left" width = "40%">
  <img src="./figures/CNN1_model_loss.png" align="left" width = "40%">
  <img src="./figures/CNN2_model_accuray.png" align="left" width = "40%">
  <img src="./figures/CNN2_model_loss.png" align="left" width = "40%">
  <img src="./figures/CNN3_batch32_model_accuray.png" align="left" width = "40%">
  <img src="./figures/CNN3_batch32_model_loss.png" align="left" width = "40%"> 
  <img src="./figures/CNN4_batch32_w_img_seg_model_accuray.png" align="left" width = "40%">
  <img src="./figures/CNN4_batch32_w_img_seg_model_loss.png" align="left" width = "40%">
</p>



