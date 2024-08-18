from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.api.applications import VGG16


# Set up the data generators
train_datagen = ImageDataGenerator(rescale=0.2)
val_datagen = ImageDataGenerator(rescale=0.2)
test_datagen = ImageDataGenerator(rescale=0.2)

# Load the pre-trained VGG16 model + higher level layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy after transfer learning: {test_acc}")

"""
Epoch 10/10
24/24 ━━━━━━━━━━━━━━━━━━━━ 87s 4s/step - acc: 1.0000 - loss: 0.0018 - val_acc: 0.9412 - val_loss: 0.4964
8/8 ━━━━━━━━━━━━━━━━━━━━ 21s 3s/step - acc: 0.9788 - loss: 0.2201
Test accuracy after transfer learning: 0.9734513163566589
"""