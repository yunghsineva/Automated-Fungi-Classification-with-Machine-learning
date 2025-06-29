## Automated Fungi Classification with Machine learning
**Project Goal**: Using Machine Learning Algorithm to classify fungi into ‘edible’ or ‘poisonous’ species from photos.

### Models:
**1. DINOv2 + DNN** `Automated Fungi Classification (DINOv2 + DNN).ipynb`
   - DINOv2 is a self-supervised foundation model used for extracting image features from photos. These features can then be applied to further tasks.
   - Deep Neural Network (DNN) is used as the classifier in this case. The DNN will take the labeled features extracted by DINOv2 as input, and through 3 layers, it will train and classify fungi species as either "edible" or "poisonous."

**2. ResNet18** `Automated Fungi Classification (ResNet18).ipynb`
   Resnet18 is a supervised pretraining model, its structure includes the feature extraction from the labeled data, followed by a linear classifier. 

### Data Preprocessing:
- To address the issue of class imbalance, we are using a WeightedRandomSampler.
- Some data augmentations are applied to improve the generalization.

### Performance and Evaluation (DINOv2 + DNN vs. ResNet18)
![image](https://github.com/user-attachments/assets/ead573d4-abea-46f1-94df-44d2aeb4a73c)
![image](https://github.com/user-attachments/assets/5acc1616-62ce-495e-a165-088b1da052b7)

### Recommended Model Architecture: DINOv2 + Deep Neural Network (DNN)
- The Dino + Deepnet model outperforms the Resnet model on all major performance metrics, including precision, recall, and ground truth/false positive rate. 
The confusion matrix of the Dino + Deepnet model shows that it classifies the "edible" and "other" categories more accurately, with a lower false positive rate. 
- The Resnet model shows an obvious overconfidence problem, which may be due to overfitting or the probability output of the model is not properly calibrated. Although the Dino + Deepnet model also shows high confidence, its confidence calibration curve is closer to the ideal curve, indicating that its degree of overconfidence is low, and the confidence of the model prediction is more consistent with its accuracy.

### Limitation:
- Slight Overfitting: As shown in the loss and accuracy plots, there is a small gap between the training and validation curves, indicating potential overfitting to the training data.
- Calibration: From the confidence calibration curve, the model is slightly overconfident, meaning it may overestimate the confidence of its predictions.
- Data imbalance: Although I have used the WeightedRandomSampler to account for imbalance data, but if the numbers of data in each speies are balance in the beginning, the performance can be more correct.

