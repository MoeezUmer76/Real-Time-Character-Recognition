# Real-Time Character Recognition

This project implements a neural network for real-time recognition of handwritten digits (0-9) and letters (A-Z) using the EMNIST ByClass dataset. It features a multi-layer perceptron (MLP) trained on EMNIST data and integrates OpenCV for live webcam-based character recognition.
<br><br>
Features<br>
Neural network with customizable layers (default: 784-128-64-47).<br>
Supports EMNIST ByClass dataset (47 classes: 0-9, A-Z, some lowercase).<br>
Real-time character recognition using webcam feed.<br>
Modular code structure with separate model and main scripts.<br><br>
Prerequisites<br>
Python 3.8+<br>
Dependencies listed in requirements.txt<br>
EMNIST ByClass dataset (emnist-byclass.mat) in the data/ folder<br>
Download from Kaggle or NIST<br>
Webcam for live predictions
