# Plant Leaf Disease Classifier

A deep learning project that identifies plant leaf diseases from a photo. It uses a Convolutional Neural Network (CNN), MobileNetV2, with transfer learning to classify leaves into 38 classes across 14 crops. The app shows the crop, the disease, a healthy or diseased verdict, the top 3 predictions and general care advice, and it uses Grad-CAM heatmaps to show what the model looked at.

> This is a prediction tool for learning and demonstration. It is not a professional diagnosis.

## Features
- CNN-based image classification with MobileNetV2 transfer learning
- 38 classes covering 14 crops, including healthy and diseased leaves
- Fine-tuning of the top layers of the network after the initial training
- Evaluation with a confusion matrix and a per-class classification report
- Grad-CAM heatmaps that show which part of the leaf influenced the prediction
- Top 3 predictions with a low-confidence warning
- General care advice for each detected disease
- Gradio web app: upload a leaf photo and get the result

## Dataset
Kaggle: [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- About 87.9K images, 38 classes, 14 crops
- Training set: 70,295 images
- Validation set: 17,572 images
- Test set: 33 images
- Each folder is named Crop___Condition, so the folder name is the label

## Model
The model uses a pretrained MobileNetV2 CNN as a feature extractor.

Input 224x224x3 -> MobileNetV2 (ImageNet weights) -> GlobalAveragePooling2D -> Dropout (0.2) -> Dense (38, softmax)

Training happens in two stages:
1. **Transfer learning.** The MobileNetV2 backbone is frozen and only the new classification head (48,678 parameters out of 2,306,662) is trained, using Adam (learning rate 1e-3) and categorical cross-entropy for 3 epochs.
2. **Fine-tuning.** The top layers of MobileNetV2 are unfrozen and trained with a much lower learning rate to adapt the features to leaf images.

Everything was trained in Google Colab on a T4 GPU with TensorFlow and Keras.

## Results
| Stage | Validation accuracy |
|---|---|
| Transfer learning (frozen backbone, 3 epochs) | 95.5% |

- Validation: 17,572 images checked after each epoch. They are not used to update the weights.
- Test: 33 separate images run one by one after training. 31 of 33 were correct (93.9%).
- The two test mistakes were AppleScab3 (predicted as Squash powdery mildew) and TomatoEarlyBlight3 (predicted as Tomato target spot).
- Tomato early blight gave the lowest confidence, even when the prediction was correct.

## Evaluation
The confusion matrix and the classification report (precision, recall and F1 for every class) are computed on the 17,572 validation images. Overall accuracy can hide weak classes, so the per-class results show where the model struggles, mainly among visually similar tomato diseases.

## Explainability with Grad-CAM
Grad-CAM produces a heatmap over the leaf image that shows which regions pushed the model toward its prediction. This is used to check that the model focuses on the leaf and its lesions and not on the background.

## Demo App
The Gradio app lets a user upload a leaf photo. It returns:
- the crop and the disease name
- a HEALTHY or DISEASED verdict
- the confidence and the top 3 predictions
- a Grad-CAM heatmap
- general care advice for the detected disease
- a warning to retake the photo when confidence is below 70%

Run it locally:
```
pip install -r requirements.txt
python app.py
```

## Limitations
- The model knows only 38 classes. Any other leaf still receives one of these answers.
- Training photos are clean, single leaves on plain backgrounds, so photos taken in the field are less reliable.
- High confidence does not guarantee a correct answer. One wrong test prediction had 91% confidence.
- The dataset is augmented, so copies of the same image may appear in both the training and validation sets, which can make validation accuracy look slightly higher than it truly is.
- Care advice is general information, not a replacement for an agronomist.

## Tech Stack
Python, TensorFlow, Keras, MobileNetV2, NumPy, Matplotlib, Gradio, Google Colab
