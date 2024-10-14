from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import layers, models

def logistic_regression_model():
    return LogisticRegression(max_iter=1000)

def mlp_model():
    return MLPClassifier(hidden_layer_sizes=(512,), max_iter=100, solver='adam', random_state=42)

def cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
