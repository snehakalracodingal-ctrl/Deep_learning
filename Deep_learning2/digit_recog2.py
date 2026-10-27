

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print('The model has successfully trained')

model.save('mnist.keras')
print('Saving the model as mnist.keras')



score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

from google.colab import files

uploaded = files.upload()
filename = list(uploaded.keys())[0]


def load_image(filename):
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    if img.mean() > 127:
        img = 255 - img
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    return img

def run_example():
    img = load_image(filename)
    loaded_model = load_model('mnist.keras')
    predict_value = loaded_model.predict(img)
    digit = argmax(predict_value)
    print('Predicted digit:', digit)

run_example()