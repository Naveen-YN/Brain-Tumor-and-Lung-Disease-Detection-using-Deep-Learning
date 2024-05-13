# Brain Tumor and Lung Disease Detection using Deep Learning

This repository contains an application that utilizes deep learning models for the classification of medical images into various categories of diseases. Currently, it supports the detection of brain tumors and lung diseases. The user interface is developed using PyQt5.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- PyQt5
- TensorFlow
- PIL (Python Imaging Library)
- NumPy

You can install these dependencies using pip:

```bash
pip install PyQt5 tensorflow pillow numpy
```

**Note: This code is designed to run without errors only in Python 3.11.6 version.**

## Datasets

To train the models, you need to download the following datasets from Kaggle:

1. [Brain Tumor Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
2. [Lung Disease Dataset](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types)

## Pre-trained models

If its hard to download datasets then you can download and use below pre-trainined models and load them in code by updating paths:
- [Brain Tumor and Lung Disease Pre-Trained Models](https://drive.google.com/drive/folders/1IgzyR1LaNPm9pStxJ8r_EXdWehxzyd-u?usp=sharing)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Naveen-YN/Brain-tumor-and-lung-disease-detection-using-deep-learning.git
```

2. Navigate to the project directory:

```bash
cd brain-tumor-lung-disease-detection
```

3. Update the directory paths in the model training code according to the location of your downloaded datasets from Kaggle.

4. Run the application:

```bash
python main.py
```

5. Follow the instructions on the user interface to upload and classify medical images.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.


