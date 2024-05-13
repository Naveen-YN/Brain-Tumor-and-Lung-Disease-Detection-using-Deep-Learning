```markdown
# Brain Tumor and Lung Disease Detection using Deep Learning

This repository contains an application that utilizes deep learning models for the classification of medical images into various categories of diseases. Currently, it supports the detection of brain tumors and lung diseases. The user interface is developed using PyQt5.

## Installation

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

If it's hard to download datasets, you can download and use the pre-trained models available in the link below:

- [Brain Tumor and Lung Disease Pre-Trained Models](https://drive.google.com/drive/folders/1IgzyR1LaNPm9pStxJ8r_EXdWehxzyd-u?usp=sharing)

## Usage

### Running the Application

Run the `app.py` file to launch the application.

### Training the Models (Optional - Run these files if you downloaded datasets otherwise you can download and load peretrained models)

#### Brain Tumor Detection Model

Run the `Brain Tumor Training.py` file to train the brain tumor detection model.

#### Lung Disease Detection Model

Run the `Lung Disease Training.py` file to train the lung disease detection model.

```bash
python "Brain Tumor Training.py"
python "Lung Disease Training.py"
```

Make sure to update the file paths in the training scripts before running them.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
