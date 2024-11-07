import tensorflow as tf
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from google.colab import files

# Get MNIST data and cache it
def get_mnist_data():
    path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=path)
    return (x_train, y_train, x_test, y_test)

# Train model with MNIST data for exactly 20 epochs
def train_model(x_train, y_train, x_test, y_test):
    # Normalize data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train for 20 epochs without early stopping
    model.fit(x_train, y_train, epochs=20, verbose=1)
    return model

# Predict digit using the model
def predict(model, img):
    img = np.array([img])
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

# Main function to upload an image for prediction in Colab
def main():
    # Load or train model
    try:
        model = tf.keras.models.load_model('model.keras')
        print('Loaded saved model.')
    except:
        print("Training model for 20 epochs...")
        (x_train, y_train, x_test, y_test) = get_mnist_data()
        model = train_model(x_train, y_train, x_test, y_test)
        model.save('model.keras')

    # Upload image
    uploaded = files.upload()
    for filename in uploaded.keys():
        print('Uploaded file:', filename)

        # Read the uploaded image
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # Preprocess the image: Resize to 28x28, invert colors, and normalize
        img_resized = cv2.resize(img, (28, 28))
        img_inverted = cv2.bitwise_not(img_resized)
        img_normalized = img_inverted / 255.0

        # Display the processed image in Colab
        cv2_imshow(img_normalized * 255)

        # Predict the digit and display result
        result = predict(model, img_normalized)
        print("Predicted Digit:", result)

# Run the main function
if __name__ == '__main__':
    main()
