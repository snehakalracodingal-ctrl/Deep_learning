
 
import numpy as np
import matplotlib.pyplot as plt
 
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)
 

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title("Digit: " + str(y_train[i]))
    plt.axis("off")
plt.show()
 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32") / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32") / 255
 
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
input_shape = (28, 28, 1)
 

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
])
 
optimizer = SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()
 

history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    validation_data=(x_test, y_test)
)
print("The model has successfully trained.")
 

model.save("mnist_digit_recognizer.h5")
print("Model saved as mnist_digit_recognizer.h5")
 
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
 

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Training Performance")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
 

saved_model = load_model("mnist_digit_recognizer.h5")
predictions = saved_model.predict(x_test)
 
plt.figure(figsize=(12, 5))
for i in range(10):
    predicted_digit = np.argmax(predictions[i])
    actual_digit = np.argmax(y_test[i])
    print("Image", i + 1, "- Predicted:", predicted_digit, "| Actual:", actual_digit)
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("P: " + str(predicted_digit) + " | A: " + str(actual_digit))
    plt.axis("off")
plt.show()
 
def predict_custom_digit(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    img = img_to_array(img).reshape(1, 28, 28, 1).astype("float32") / 255
    prediction = saved_model.predict(img)
    print("Predicted digit:", np.argmax(prediction))
 
