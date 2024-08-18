from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from keras.api.applications import VGG16
from keras.api.models import Model
import matplotlib.pyplot as plt
import requests
import zipfile
import os


# URL of the dataset
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
def download_dataset(url):
    response = requests.get(url)
    zip_file_path = 'cats_and_dogs_filtered.zip'

    # Save the file locally
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)

    # Extract the downloaded file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(zip_file_path)


# Define paths
base_dir = 'cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


train_datagen = ImageDataGenerator(rescale=0.2)
validation_datagen = ImageDataGenerator(rescale=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Load VGG16 pre-trained model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=x)
model.summary()

# compile model
opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

# Train the model
history = model.fit(
      train_generator,
      epochs=10,
      validation_data=validation_generator
    )


# Plotting the results
def plot_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], label='train_accuracy')
    plt.plot(history.history['val_acc'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


plot_history(history)

"""
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
2024-07-14 00:24:09.817383: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "functional"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ input_layer (InputLayer)        │ (None, 224, 224, 3)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv1 (Conv2D)           │ (None, 224, 224, 64)   │         1,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_conv2 (Conv2D)           │ (None, 224, 224, 64)   │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block1_pool (MaxPooling2D)      │ (None, 112, 112, 64)   │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv1 (Conv2D)           │ (None, 112, 112, 128)  │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_conv2 (Conv2D)           │ (None, 112, 112, 128)  │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block2_pool (MaxPooling2D)      │ (None, 56, 56, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv1 (Conv2D)           │ (None, 56, 56, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv2 (Conv2D)           │ (None, 56, 56, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_conv3 (Conv2D)           │ (None, 56, 56, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block3_pool (MaxPooling2D)      │ (None, 28, 28, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv1 (Conv2D)           │ (None, 28, 28, 512)    │     1,180,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv2 (Conv2D)           │ (None, 28, 28, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_conv3 (Conv2D)           │ (None, 28, 28, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block4_pool (MaxPooling2D)      │ (None, 14, 14, 512)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv1 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv2 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_conv3 (Conv2D)           │ (None, 14, 14, 512)    │     2,359,808 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ block5_pool (MaxPooling2D)      │ (None, 7, 7, 512)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 25088)          │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 512)            │    12,845,568 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 512)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 1)              │           513 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 27,560,769 (105.14 MB)
 Trainable params: 12,846,081 (49.00 MB)
 Non-trainable params: 14,714,688 (56.13 MB)
Epoch 1/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 277s 4s/step - acc: 0.8534 - loss: 0.9674 - val_acc: 0.9720 - val_loss: 0.1259
Epoch 2/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 247s 4s/step - acc: 0.9776 - loss: 0.0704 - val_acc: 0.9620 - val_loss: 0.1353
Epoch 3/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 257s 4s/step - acc: 0.9850 - loss: 0.0506 - val_acc: 0.9750 - val_loss: 0.1104
Epoch 4/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 254s 4s/step - acc: 0.9982 - loss: 0.0093 - val_acc: 0.9730 - val_loss: 0.1188
Epoch 5/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 210s 3s/step - acc: 0.9964 - loss: 0.0119 - val_acc: 0.9710 - val_loss: 0.1240
Epoch 6/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 210s 3s/step - acc: 1.0000 - loss: 0.0019 - val_acc: 0.9720 - val_loss: 0.1221
Epoch 7/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 206s 3s/step - acc: 0.9998 - loss: 0.0017 - val_acc: 0.9720 - val_loss: 0.1215
Epoch 8/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 215s 3s/step - acc: 0.9994 - loss: 0.0030 - val_acc: 0.9720 - val_loss: 0.1276
Epoch 9/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 234s 4s/step - acc: 0.9977 - loss: 0.0041 - val_acc: 0.9730 - val_loss: 0.1302
Epoch 10/10
63/63 ━━━━━━━━━━━━━━━━━━━━ 213s 3s/step - acc: 1.0000 - loss: 0.0010 - val_acc: 0.9730 - val_loss: 0.1306

"""