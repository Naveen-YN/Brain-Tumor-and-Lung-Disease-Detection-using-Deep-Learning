from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QWidget
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained models for Brain Tumor and Lung Disease detection
brain_tumor_model = tf.keras.models.load_model('Trained Model/brain_tumor_model_best.h5')
lung_disease_model = tf.keras.models.load_model('Trained Model/lung_disease_model_best.h5')


# Define class names for brain tumor and lung disease
class_names_brain_tumor = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary Tumor'}
class_names_lung_disease = {0: 'Bacterial Pneumonia', 1: 'Corona Virus Disease', 2: 'Normal', 3: 'Tuberculosis', 4: 'Viral Pneumonia'}

# Define curability information for diseases
curability_info = {
    'Glioma': {'curable': False, 'treatment': 'Surgery, radiation therapy, chemotherapy'},
    'Meningioma': {'curable': True, 'treatment': 'Surgery'},
    'No Tumor': {'curable': True, 'treatment': 'N/A'},
    'Pituitary Tumor': {'curable': True, 'treatment': 'Surgery, medication'},
    'Bacterial Pneumonia': {'curable': True, 'treatment': 'Antibiotics'},
    'Corona Virus Disease': {'curable': False, 'treatment': 'Supportive care, vaccination'},
    'Normal': {'curable': True, 'treatment': 'N/A'},
    'Tuberculosis': {'curable': True, 'treatment': 'Antibiotics, anti-TB drugs'},
    'Viral Pneumonia': {'curable': True, 'treatment': 'Antiviral medications, supportive care'}
}

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("designer.ui", self)

        # Connect signals and slots for Lung Disease tab
        self.tabWidget.findChild(QWidget, "select_button_lung").clicked.connect(self.open_file_lung)
        self.tabWidget.findChild(QWidget, "detect_button_lung").clicked.connect(self.detect_disease_lung)

        # Connect signals and slots for Brain Tumor tab
        self.tabWidget.findChild(QWidget, "select_button_brain").clicked.connect(self.open_file_brain)
        self.tabWidget.findChild(QWidget, "detect_button_brain").clicked.connect(self.detect_disease_brain)

        # Initialize variables to store image arrays
        self.img_array_lung = None
        self.img_array_brain = None

    def open_file_lung(self):
        self.file_path_lung, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if self.file_path_lung:
            img = Image.open(self.file_path_lung)

            # Convert grayscale images to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Get canvas dimensions
            canvas_width = self.canvas_lung.width()
            canvas_height = self.canvas_lung.height()

            # Calculate aspect ratio
            img_aspect_ratio = img.width / img.height
            canvas_aspect_ratio = canvas_width / canvas_height

            # Resize image to fit the canvas while maintaining aspect ratio
            if img_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect_ratio)
            else:
                new_width = int(canvas_height * img_aspect_ratio)
                new_height = canvas_height

            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.img_array_lung = np.array(img)

            # Converts PIL Image to QImage
            height, width = self.img_array_lung.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.img_array_lung.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas_lung.setPixmap(pixmap)

            # Enable detect button
            self.detect_button_lung.setEnabled(True)

    def open_file_brain(self):
        self.file_path_brain, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if self.file_path_brain:
            img = Image.open(self.file_path_brain)

            # Convert grayscale images to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Get canvas dimensions
            canvas_width = self.canvas_brain.width()
            canvas_height = self.canvas_brain.height()

            # Calculate aspect ratio
            img_aspect_ratio = img.width / img.height
            canvas_aspect_ratio = canvas_width / canvas_height

            # Resize image to fit the canvas while maintaining aspect ratio
            if img_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect_ratio)
            else:
                new_width = int(canvas_height * img_aspect_ratio)
                new_height = canvas_height

            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.img_array_brain = np.array(img)
 
            # Converts PIL Image to QImage
            height, width = self.img_array_brain.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.img_array_brain.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas_brain.setPixmap(pixmap)

            # Enable detect button
            self.detect_button_brain.setEnabled(True)
    def detect_disease_lung(self):
        if self.img_array_lung is not None:
            try:
                image = Image.fromarray(self.img_array_lung)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                predictions = lung_disease_model.predict(image)
                class_names = class_names_lung_disease

                predicted_class_index = np.argmax(predictions)
                predicted_class = class_names.get(predicted_class_index, 'Unknown')

                # Add information about lung disease to result_label_1
                result_text_1 = f"Disease: {predicted_class}\n"

                # Add curability information if available to result_label_1
                curability = curability_info.get(predicted_class)
                if curability:
                    result_text_1 += f"Curable: {'Yes' if curability['curable'] else 'No'}\n"
                    result_text_1 += f"Treatment: {curability['treatment']}"

                self.result_label_lung.setText(result_text_1)

                # Add small paragraph about lung disease to result_label_2
                result_text_2 = self.get_disease_info_paragraph(predicted_class)
                self.result_label_lung_2.setText(result_text_2)

            except Exception as e:
                print("Error occurred during lung disease detection:", str(e))
                QMessageBox.warning(self, "Error", "Error occurred during lung disease detection.")
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image in the lung tab first.")

    def detect_disease_brain(self):
        if self.img_array_brain is not None:
            try:
                image = Image.fromarray(self.img_array_brain)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                predictions = brain_tumor_model.predict(image)
                class_names = class_names_brain_tumor

                predicted_class_index = np.argmax(predictions)
                predicted_class = class_names.get(predicted_class_index, 'Unknown')

                # Add information about brain tumor to result_label_1
                result_text_1 = f"Disease: {predicted_class}\n"

                # Add curability information if available to result_label_1
                curability = curability_info.get(predicted_class)
                if curability:
                    result_text_1 += f"Curable: {'Yes' if curability['curable'] else 'No'}\n"
                    result_text_1 += f"Treatment: {curability['treatment']}"

                self.result_label_brain.setText(result_text_1)

                # Add small paragraph about brain tumor to result_label_2
                result_text_2 = self.get_disease_info_paragraph(predicted_class)
                self.result_label_brain_2.setText(result_text_2)

            except Exception as e:
                print("Error occurred during brain tumor detection:", str(e))
                QMessageBox.warning(self, "Error", "Error occurred during brain tumor detection.")
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image in the brain tab first.")


    def get_disease_info_paragraph(self, disease):
        # Add your small paragraph about the disease here
        if disease == 'Bacterial Pneumonia':
            return ("Bacterial pneumonia is a type of lung infection caused by bacteria. "
                    "It often presents with symptoms such as fever, cough, chest pain, and difficulty breathing. "
                    "Antibiotics are commonly used for treatment. "
                    "If left untreated, it can lead to serious complications, especially in older adults and people with weakened immune systems.")
        elif disease == 'Corona Virus Disease':
            return ("Coronavirus disease (COVID-19) is a highly contagious respiratory illness caused by the SARS-CoV-2 virus. "
                    "It can lead to a range of symptoms from mild respiratory issues to severe pneumonia and acute respiratory distress syndrome (ARDS). "
                    "In addition to respiratory symptoms, COVID-19 may also cause fatigue, muscle or body aches, loss of taste or smell, sore throat, and headache. "
                    "Treatment involves supportive care, including rest, fluids, and fever-reducing medication. "
                    "In severe cases, hospitalization and oxygen therapy may be necessary. "
                    "Vaccination is recommended to prevent COVID-19 infection and reduce its spread.")
        elif disease == 'Glioma':
            return ("Glioma is a type of brain tumor that arises from glial cells in the brain. "
                    "It can be aggressive and difficult to treat. "
                    "Symptoms of glioma depend on the tumor's location and size and may include headaches, seizures, nausea, vomiting, and changes in vision or memory. "
                    "Treatment options include surgery, radiation therapy, and chemotherapy. "
                    "However, the prognosis for glioma patients varies depending on several factors, including tumor grade and molecular characteristics.")
        elif disease == 'Meningioma':
            return ("Meningioma is a type of brain tumor that develops from the meninges, the protective membranes surrounding the brain and spinal cord. "
                    "Most meningiomas are benign and slow-growing. "
                    "Symptoms of meningioma may include headaches, seizures, weakness or numbness in the limbs, changes in vision, and cognitive changes. "
                    "Surgery is the primary treatment for meningioma, although radiation therapy may be used in some cases. "
                    "The outlook for patients with meningioma is generally favorable, especially for those with benign tumors.")
        elif disease == 'No Tumor':
            return ("No tumor detected in the brain. This result indicates the absence of abnormal growths in the brain. "
                    "However, it's essential to continue monitoring for any changes in symptoms or new developments.")
        elif disease == 'Normal':
            return ("The lung scan appears normal with no signs of infection or abnormalities. "
                    "Maintaining good respiratory health through regular exercise, a balanced diet, and avoiding exposure to pollutants can help prevent lung diseases.")
        elif disease == 'Pituitary Tumor':
            return ("A pituitary tumor is a growth of abnormal cells in the pituitary gland, which is located at the base of the brain. "
                    "Depending on its size and hormone-secreting properties, treatment may involve surgery or medication. "
                    "Symptoms of pituitary tumors may include headaches, vision problems, fatigue, weight gain or loss, and hormonal imbalances. "
                    "The prognosis for pituitary tumor patients depends on various factors, including tumor size, hormone levels, and response to treatment.")
        elif disease == 'Tuberculosis':
            return ("Tuberculosis (TB) is a bacterial infection that primarily affects the lungs but can also affect other parts of the body. "
                    "It spreads through the air when an infected person coughs or sneezes. "
                    "Symptoms of TB may include coughing up blood, chest pain, fatigue, fever, night sweats, and unexplained weight loss. "
                    "Treatment involves a combination of antibiotics and anti-TB drugs taken over several months. "
                    "Early diagnosis and treatment are crucial to prevent the spread of TB and reduce complications.")
        elif disease == 'Viral Pneumonia':
            return ("Viral pneumonia is caused by viruses such as influenza, respiratory syncytial virus (RSV), or the coronavirus. "
                    "It typically presents with symptoms similar to those of bacterial pneumonia, including fever, cough, chest pain, and difficulty breathing. "
                    "Treatment involves antiviral medications, rest, and supportive care, such as staying hydrated and getting plenty of rest. "
                    "In severe cases, hospitalization and oxygen therapy may be necessary. "
                    "Vaccination against influenza and other preventable viral infections can help reduce the risk of viral pneumonia.")
        else:
            return "Information about this disease is not available."



if __name__ == "__main__":
    app = QApplication([])
    main_window = App()
    main_window.show()
    app.exec_()