# Number Prediction AI

This project is focused on developing a neural network model to recognize handwritten digits from 0 to 9. Using a dataset of 5000 training images, the model is trained to accurately predict the digit that is drawn on a simple drawing application.

## Project Overview

The Number Prediction AI project involves 3 steps:

1. **Data Collection**: The model is trained on a dataset of 5000 images, where each image corresponds to a digit from 0 to 9. This dataset teaches the neural network how to identify different digits based on their shape.

2. **Model Creation**: After data preprocessing, a neural network is constructed using TensorFlow and trained on the dataset. The network learns the patterns and characteristics of each digit, enabling it to make predictions when presented with new, unseen images.

3. **Prediction Application**: A user-friendly drawing interface is developed, allowing users to draw a digit. The trained model then predicts the digit based on the input from the drawing app.

## Features

- **Handwritten Digit Recognition**: The neural network can recognize digits from 0-9 with high accuracy.
- **Interactive Drawing App**: Users can draw a digit on the screen, and the model will predict which digit was drawn.
- **Model Persistence**: The trained model can be saved and loaded for future predictions without needing to retrain.

## Screenshots

### Prediction Application Interface
![Prediction Application Interface](Fig2.png)

### Model Training Process
![Model Training Process](Fig1.png)

## Getting Started

Follow these steps to set up the project on your local machine.

### Prerequisites

Ensure you have Python installed, along with the following dependencies:
- NumPy
- TensorFlow

You can install the required dependencies using pip:

```bash
pip install numpy tensorflow matplotlib ipywidgets Pillow
```

You would also need python-tk
```bash
brew install python-tk
```

### Running the Project

Train the Model: To train the neural network, run the following command:

```bash
python create_model.py
```

This script will load the dataset, preprocess the images, build the neural network, and train it on the provided data.

Load and Use the Model: After training, you can load the saved model and use it for predictions:

```bash
python load_model.py
```

This script will load the trained model and launch the drawing app, where you can draw digits and see the model's predictions in real-time.

## Contributing
Contributions are welcome! If you have any suggestions, bug fixes, or new features to add, feel free to open a pull request.

## Author
Naman Gupta - Initial work
