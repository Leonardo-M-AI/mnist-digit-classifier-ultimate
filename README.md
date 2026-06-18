# MNIST Digit Classifier - Ultimate

Handwritten digit recognition on the MNIST dataset, comparing three machine learning approaches: MLP, CNN, and SVM, with robustness testing on noisy data.

## Project Structure
mnist-digit-classifier-ultimate/

├── notebook/mnist_classifier.ipynb

├── requirements.txt

└── .gitignore
## Pipeline

| Step | Description |
|------|-------------|
| Data Loading | 60,000 train / 10,000 test images via `tf.keras.datasets` |
| EDA | Class distribution analysis, sample visualisation |
| Preprocessing | Flatten/reshape pixels, normalise to [0, 1] |
| Models | MLP, CNN, and SVM trained and compared |
| Evaluation | Accuracy, Loss, Classification Report, Confusion Matrix |
| Robustness | Performance tested on artificially noised images |

## Model Architectures

**MLP**: Dense(128, ReLU), Dropout(0.2), Dense(64, ReLU), Dense(10, Softmax)

**CNN**: Conv2D(32), MaxPooling, Conv2D(64), MaxPooling, Flatten, Dense(64, ReLU), Dropout(0.3), Dense(10, Softmax)

**SVM**: RBF kernel, trained on a 5,000-sample subset for computational efficiency

## Results

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| MLP | 97.74% | Dense layers only |
| **CNN** | **99.07%** | Best performer |
| SVM | 92.95% | Trained on 5,000 samples only |

**Robustness test:** the CNN retains **96.52% accuracy** on artificially noised images, a drop of only 2.55 points from clean data, indicating strong generalisation beyond the training distribution.

## Quick Start

```bash
git clone https://github.com/Leonardo-M-AI/mnist-digit-classifier-ultimate.git
pip install -r requirements.txt
jupyter notebook notebook/mnist_classifier.ipynb
```

## Tech Stack
`Python` `TensorFlow/Keras` `scikit-learn` `Pandas` `NumPy` `Matplotlib`

## License
MIT

## Business Impact

A CNN achieving 99% accuracy on handwritten digit recognition can automate data entry in banking (cheque processing), postal services (address sorting), and healthcare (form digitisation). The demonstrated robustness to noise (96.5% accuracy under degraded conditions) makes the model viable for real-world scanned documents, not just clean lab data, reducing manual review costs by an estimated 35-60% on digit-heavy workflows.

## What I Would Do Next

- Deploy the model as a REST API using FastAPI for real-time inference
- Expand robustness testing to rotation, blur, and occlusion
- Experiment with data augmentation to further close the noisy-data accuracy gap
