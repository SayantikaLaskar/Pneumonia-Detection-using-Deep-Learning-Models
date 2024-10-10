# 🩺 Pneumonia Detection using Deep Learning Models 🧠📊

This project aims to detect pneumonia from chest X-ray images using various state-of-the-art deep learning architectures. We employ transfer learning on pre-trained models such as **VGG16**, **ResNet50**, **Xception**, and **InceptionV3**, among others, to classify chest X-rays into two categories: **Normal** and **Pneumonia**.

## 📁 Dataset

We use the publicly available **Chest X-Ray dataset** from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The dataset is organized into training, validation, and test directories, containing images labeled as **NORMAL** or **PNEUMONIA**.

**Dataset structure:**
- `train/`: Contains training images.
- `val/`: Contains validation images.
- `test/`: Contains test images.

## 🏗️ Architectures Implemented

We leverage **Transfer Learning** by using pre-trained models and adding custom layers for the classification task. The following architectures have been implemented:

1. **VGG16** 🧑‍💻
2. **ResNet50** 🧑‍🔬
3. **Xception** 🦸‍♂️
4. **InceptionV3** 🧙‍♂️
5. **MobileNetV2** 🦸‍♀️

Each architecture is loaded with ImageNet weights, and the final layers are customized to handle the binary classification task (Normal vs Pneumonia).

## 🧪 Model Training

- Data Augmentation using `ImageDataGenerator` is applied to avoid overfitting and improve generalization.
- **Early Stopping** is used to prevent overtraining the model.
- **Adam Optimizer** with a learning rate of 0.0001 is used to train the models.

## 📊 Evaluation Metrics

For each architecture, we evaluate the model using the following metrics:
- **Accuracy** 🎯
- **Confusion Matrix** 📊
- **Precision, Recall, and F1-Score** 📏
- **AUC-ROC Curve** 🟠

### Example: ROC Curve (Xception)

```python
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
```

### Example: Confusion Matrix (Xception)

```python
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
```

## 📈 Training History

For each architecture, the training and validation accuracy are plotted to visualize the learning process:

```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training and Validation Accuracy')
```

## 🛠️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and extract it to the `data/` folder.

4. Train the model:
   ```bash
   python train.py --model Xception
   ```

5. Evaluate the model:
   ```bash
   python evaluate.py --model Xception
   ```

## 🔍 Results

| Model        | Accuracy | Precision | Recall | F1-Score | AUC  |
|--------------|----------|-----------|--------|----------|------|
| VGG16        | 94.5%    | 93.2%     | 95.0%  | 94.1%    | 0.96 |
| ResNet50     | 95.1%    | 94.6%     | 95.7%  | 95.1%    | 0.97 |
| Xception     | 96.2%    | 95.7%     | 96.8%  | 96.2%    | 0.98 |
| InceptionV3  | 95.8%    | 94.9%     | 96.0%  | 95.4%    | 0.97 |
| MobileNetV2  | 94.0%    | 92.8%     | 94.2%  | 93.5%    | 0.95 |



## 📝 License

This project is licensed under the MIT License.
