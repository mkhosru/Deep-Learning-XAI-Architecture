Interpretable Deep Learning for European Architectural Heritage Classification
 Using XAI

This project focuses on classifying European architectural heritage elements using deep learning, with an emphasis on explainability. We compare three models — ResNet50, tf_efficientnetv2_s, and ViT Base16 — and use Grad-CAM and LIME to visualize how the models make their predictions.

Our findings show that ResNet50 (pretrained on ImageNet) delivers the best performance, achieving the highest accuracy and most consistent per-class results. The Predicted Probabilities for sample test images confirm its confidence (e.g., 99.99% for the Column class), and the explainability visualizations verify that the model is focusing on meaningful architectural features.

📂 Dataset

You can access the dataset from:

Google Drive: https://drive.google.com/drive/u/0/folders/1poNByex0IIZVw4MLj5bwMJ0TQdDNoZ-f

Kaggle: https://www.kaggle.com/datasets/ikobzev/architectural-heritage-elements-image64-dataset

The dataset contains images from 10 architectural classes:
Altar, Apse, Bell Tower, Column, Inner Dome, Outer Dome, Flying Buttress, Gargoyle, Stained Glass, Vault

🧠 Models

We trained and compared the following models:

ResNet50 (CNN) – Pretrained and from scratch

tf_efficientnetv2_s (CNN) – Pretrained and from scratch

ViT Base16 (Vision Transformer) – Pretrained and from scratch

All models were trained for 10 epochs using AdamW optimizer, with learning rate 2e-4, weight decay 1e-4, and data augmentation (random flip, rotation, color jitter).

📊 Results
Model (Pretrained)	Test Accuracy	Precision	Recall	F1-score
ResNet50	Highest, Best overall performance, Most consistent per-class results	
tf_efficientnetv2_s	Very close	Well-balanced	Stable	
ViT Base16	Slightly lower	Good generalization	Sensitive to data size	

ResNet50 consistently achieved top performance, with very confident predictions (e.g., 99.99% for Column class).

🖼️ Explainable AI

To make the model’s decision-making transparent, we used:

Grad-CAM: Highlights the most important regions in the image (e.g., columns, stained glass).

LIME: Outlines superpixels contributing most to the classification decision.

These visualizations confirmed that the model attends to meaningful features, increasing trust in its predictions.

🚀 How to Use

Clone this repository:

git clone https://github.com/yourusername/architectural-heritage-xai.git
cd architectural-heritage-xai


Install dependencies:

pip install -r requirements.txt


Run training:

python train.py --model resnet50 --pretrained True


Test on a single image:

python predict.py --image path/to/image.jpg --model resnet50


View Grad-CAM and LIME visualizations generated in the /results folder.

📜 License

This project is for research and educational purposes. The dataset is credited to Ikobzev on Kaggle
. Please review the dataset license before commercial use.
