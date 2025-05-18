import numpy as np
import cv2
from scipy.io import loadmat
import os
from model import *


def preprocess_frame_for_emnist(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        img_cropped = img_thresh[y:y + h, x:x + w]
    else:
        img_cropped = img_thresh
    img_resized = cv2.resize(img_cropped, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(img_resized, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    img_normalized = padded / 255.0
    img_flattened = img_normalized.reshape(1, -1)
    return img_flattened


def main():
    model_file = 'models/trained_emnist_weights.npy'

    # Define label map
    label_map = {i: str(i) if i <= 9 else chr(ord('A') + i - 10) for i in range(47)}

    # Load or train model
    if os.path.exists(model_file):
        print("Loading saved model...")
        model = np.load(model_file, allow_pickle=True).item()
        weights = model['weights']
        biases = model['biases']
    else:
        print("Loading EMNIST ByClass dataset...")
        emnist = loadmat('data/emnist-byclass.mat')

        # Extract train/test data
        X_train = emnist['dataset']['train'][0][0]['images'][0][0]
        y_train = emnist['dataset']['train'][0][0]['labels'][0][0].flatten()
        X_test = emnist['dataset']['test'][0][0]['images'][0][0]
        y_test = emnist['dataset']['test'][0][0]['labels'][0][0].flatten()

        # Normalize
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # Fix EMNIST rotation
        X_train = X_train.reshape(-1, 28, 28)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_train = X_train.reshape(-1, 784)

        X_test = X_test.reshape(-1, 28, 28)
        X_test = np.transpose(X_test, (0, 2, 1))
        X_test = X_test.reshape(-1, 784)

        # Map labels to 0-46
        unique_labels = np.unique(y_train)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_train_mapped = np.array([label_mapping[label] for label in y_train])
        y_test_mapped = np.array([label_mapping[label] for label in y_test])

        # Train
        num_classes = len(np.unique(y_train))
        layer_sizes = [784, 128, 64, num_classes]
        weights, biases, history = train(X_train, y_train_mapped, layer_sizes, epochs=15, batch_size=128,
                                         learning_rate=0.1)

        # Save model
        os.makedirs('models', exist_ok=True)
        np.save(model_file, {'weights': weights, 'biases': biases})

    # Live webcam prediction
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_img = preprocess_frame_for_emnist(frame)
        prediction_idx = predict(processed_img, weights, biases)[0]
        predicted_label = label_map.get(prediction_idx, '?')

        # Display prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Prediction: {predicted_label}', (50, 50), font, 2, (255, 0, 0), 3)
        cv2.imshow('Live Prediction - Press Q to Quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()