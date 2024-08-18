from __future__ import print_function
import keras
from keras.api.datasets import cifar10
from colorama import Fore, Style

def style_print(string, color = None):
    if color == "green":
        print(f"{Fore.GREEN}{Style.BRIGHT}"
              f"{string}"
              f"{Style.RESET_ALL}")
    elif color == "blue":
        print(f"{Fore.BLUE}{Style.BRIGHT}"
              f"{string}"
              f"{Style.RESET_ALL}")
    else:
        print(string)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Define batch sizes and number of epochs
batch_size = [4, 32, 128, 1024]
num_epochs = 10

"""
-----------------------------------------MODEL 1 + 2.1 + 2.2 + 2.3------------------------------------------------------
Model_1 is set to be the same in terms of structure except for the difference of batch size.
"""
model_1 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense((512), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

"""
-----------------------------------------------MODEL 3------------------------------------------------------------------
In both layers padding = same.
"""
model_3 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense((512), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

"""
-----------------------------------------------MODEL 4------------------------------------------------------------------
Model with architecture from the MNIST network. 
"""
model_4 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

"""
-----------------------------------------------MODEL 5 -----------------------------------------------------------------
Adding additional layers of convolution and pooling as Intel offers.
"""

model_5 = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation="relu"),
        keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense((512), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

"""
-----------------------------------------------MODEL 5 PRO--------------------------------------------------------------
A model with many layers.
"""

model_5_pro = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax'),
])

"""
--------------------------------------------RUN ALL MODELS---------------------------------------------------------
"""

models = {
    "model_1": model_1,
    "model_3": model_3,
    "model_4": model_4,
    "model_5": model_5,
    "model_5_pro": model_5_pro
}

# Stores the results of all models
results = []
for model_name, model in models.items():
    style_print(f"{model_name} start running...", color="green")
    if model_name == "model_1":
        for i in batch_size:
            print(f"{model_name} with {i} batch")
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            model.fit(x_train, y_train,
                        batch_size=i,
                        epochs=num_epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)

            score = model.evaluate(x_test, y_test, verbose=0)
            r = (f"Test loss for model with {i} batch:  {score[0]}"
                 f"Test accuracy for model with {i} batch: {score[1]}")
            results.append(r)
            style_print(f"Test loss for model with {i} batch: {score[0]}", color="blue")
            style_print(f"Test accuracy for model with {i} batch: {score[1]}", color="blue")
    else:
        if model_name.startswith("model_5"):
            num_epochs = 20
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(x_train, y_train,
                  batch_size=batch_size[1],
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)

        score = model.evaluate(x_test, y_test, verbose=0)
        r = (f"Test loss for {model_name}: {score[0]}\t |"
             f"Test accuracy for {model_name}: {score[1]}")
        results.append(r)

        style_print(f"Test loss for {model_name}: : {score[0]}", color="blue")
        style_print(f"Test accuracy for {model_name}: {score[1]}\n", color="blue")

for item in results:
    print(item)

""" ------------------------Summary of results --------------------------------
Test loss and accuracy for different batch sizes:
- Batch size 4: 
  - Loss: 1.2470
  - Accuracy: 0.5655
- Batch size 32: 
  - Loss: 1.2118
  - Accuracy: 0.6140
- Batch size 128: 
  - Loss: 1.2852
  - Accuracy: 0.6120
- Batch size 1024: 
  - Loss: 1.3660
  - Accuracy: 0.6193

Test loss and accuracy for different models:
- Model 3: 
  - Loss: 0.9912
  - Accuracy: 0.6668
- Model 4: 
  - Loss: 0.9680
  - Accuracy: 0.6746
- Model 5: 
  - Loss: 1.2154
  - Accuracy: 0.7540
- Model 5 Pro: 
  - Loss: 0.6609
  - Accuracy: 0.7773 
  



Answer A:
1. Model with Batch size 1024!
2. The execution that runs the fastest is for batch_size = 1024.
   Since it only makes 49 updates per Epoch compared to for example batch_size = 4: which makes 12500 updates per Epoch.
----------------------------------------------------------------------------------------------------------------------

Answer B: improve significantly.
for batch_size = 32:
New test loss:  0.9912      |   Old test loss: 1.2118
New test accuracy: 0.6668   |   Old test accuracy: 0.6140
----------------------------------------------------------------------------------------------------------------------

Answer C:
model_4 with batch_size 32 - Didn't improve significantly.
Test loss: 0.9680
Test accuracy: 0.6746
----------------------------------------------------------------------------------------------------------------------

Answer D:
model_5 with 25 epochs:
Test loss: 1.2154
Test accuracy: 0.7540

We built a new model that adds 2 more conventions and a pooling to the Intel model
model5_pro with 25 epochs:
Test loss: 0.6609
Test accuracy: 0.7773 - ( in Colab we reach 0.81 )
"""





