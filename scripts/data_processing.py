import numpy as np

def preprocess_data():
    # Load MNIST data from local file
    path = './data/mnist.npz'
    with np.load(path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']

    # Flatten the images for logistic regression and MLP
    x_train_flat = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255

    # For CNN, reshape the data to include the channel dimension
    x_train_cnn = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
    x_test_cnn = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

    return (x_train_flat, y_train), (x_test_flat, y_test), (x_train_cnn, x_test_cnn)

if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing completed.")
