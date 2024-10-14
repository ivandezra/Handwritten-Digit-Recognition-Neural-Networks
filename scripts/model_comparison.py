import numpy as np
import pickle
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from data_processing import preprocess_data
from model_helpers import logistic_regression_model, mlp_model, cnn_model

def train_logistic_regression(x_train, y_train):
    model = logistic_regression_model()
    model.fit(x_train, y_train)
    return model

def train_mlp(x_train, y_train):
    model = mlp_model()
    model.fit(x_train, y_train)
    return model

def train_cnn(x_train, y_train):
    model = cnn_model((28, 28, 1))
    model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=1)
    return model

def compare_models():
    # Load and preprocess data
    (x_train_flat, y_train), (x_test_flat, y_test), (x_train_cnn, x_test_cnn) = preprocess_data()
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    log_reg_model = train_logistic_regression(x_train_flat, y_train)
    y_pred_log_reg = log_reg_model.predict(x_test_flat)
    
    # Train MLP
    print("Training Multilayer Perceptron...")
    mlp_model_trained = train_mlp(x_train_flat, y_train)
    y_pred_mlp = mlp_model_trained.predict(x_test_flat)

    # Train CNN
    print("Training Convolutional Neural Network...")
    cnn_model_trained = train_cnn(x_train_cnn, y_train)
    y_pred_cnn = np.argmax(cnn_model_trained.predict(x_test_cnn), axis=1)

    # Evaluate and compare accuracy
    models_results = {
        'Logistic Regression': accuracy_score(y_test, y_pred_log_reg),
        'MLP': accuracy_score(y_test, y_pred_mlp),
        'CNN': accuracy_score(y_test, y_pred_cnn)
    }

    print("Model Comparison Results:")
    for model, accuracy in models_results.items():
        print(f"{model}: {accuracy:.2f}")
    
    # Plot the comparison
    plt.bar(models_results.keys(), models_results.values())
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.savefig('./images/comparison_plot.png')
    plt.show()

    # Save results
    with open('./models/model_comparison.pkl', 'wb') as f:
        pickle.dump(models_results, f)

if __name__ == "__main__":
    compare_models()
