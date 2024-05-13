import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define paths
train_data_dir = 'E:\\AD Projects\\Brain Tumor and Lung Disease Detection\\Dataset\\Brain Tumor\\Training'
test_data_dir = 'E:\\AD Projects\\Brain Tumor and Lung Disease Detection\\Dataset\\Brain Tumor\\Testing'
batch_size = 32
image_size = (224, 224)

# Data augmentation for training set
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # New: Random rotation
    width_shift_range=0.2,  # New: Random horizontal shift
    height_shift_range=0.2,  # New: Random vertical shift
    horizontal_flip=True
)

# Data augmentation for testing set (only rescaling)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and prepare training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  
)

# Load and prepare testing data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  
)

# Define the base model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with a ReLU activation
x = Dense(512, activation='relu')(x)  # Increased units
x = Dropout(0.5)(x)  # New: Dropout for regularization

# Add another fully-connected layer
x = Dense(256, activation='relu')(x)  

# Add the final prediction layer
predictions = Dense(4, activation='softmax')(x)  

# Combine the base model and the custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Fine-tune the last few layers of the base model
for layer in base_model.layers[:-15]:  # Fine-tune from layer 15 onwards
    layer.trainable = False

# Compile the model using categorical_crossentropy for multi-class classification
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up model checkpoint to save the best weights
checkpoint = ModelCheckpoint(
    'brain_tumor_model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Train the model with model checkpoint
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,  # Increased epochs
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size,
    callbacks=[checkpoint]
)

# Save the trained model
model.save('brain_tumor_model_final.h5')
