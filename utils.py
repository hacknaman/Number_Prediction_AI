import numpy as np
import os
from PIL import ImageGrab

def load_data():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y

def display_errors(model,X,y):
    f = model.predict(X)
    yhat = np.argmax(f, axis=1)
    idxs = np.where(yhat != y[:,0])[0]
    return(len(idxs))

def NormalizeData(X):
    m , n = X.shape

    X1 = X
    X1 = X1.astype(np.uint8)

    for i in range( m ):
        image_array = X[i]
        normalized_array = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        image_array = normalized_array.astype(np.uint8)
        image_array[image_array < 20] = 0
        image_array = image_array.reshape(20,20).T
        X1[i] = image_array.reshape((400,))

    np.save("data/X_mod.npy", X1 )

def readAndSaveImagesAsNPX():
    # Directory containing the images
    directory = 'pngData'

    # Lists to store images and labels
    images = []
    labels = []

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            # Construct the full path to the image file
            filepath = os.path.join(directory, filename)
            
            # Load the image in grayscale mode
            image = ImageGrab.open(filepath).convert('L')
            
            # Resize the image to 20x20 if needed
            image = image.resize((20, 20))
            
            # Convert the image to a NumPy array
            image_array = np.array(image)
            
            # Flatten the image array to 1D if needed (20x20 = 400 elements)
            # image_array = image_array.flatten()  # Optional, if you need 1D arrays
            
            # Append the image array to the list
            images.append(image_array)
            
            # Extract the label from the filename (e.g., '2500_5.png' -> label = 5)
            label = int(filename.split('_')[-1].split('.')[0])
            
            # Append the label to the list
            labels.append(label)

    # Convert lists to NumPy arrays
    images_array = np.array(images)
    labels_array = np.array(labels)

    # Print the shapes of the arrays
    print("Images array shape:", images_array.shape)  # Should be (num_images, 20, 20)
    print("Labels array shape:", labels_array.shape)  # Should be (num_images,)

    # Optionally save the arrays to disk
    np.save('data/X.npy', images_array)
    np.save('data/Y.npy', labels_array)


def saveImagesToPng(X, y):

    m , n = X.shape
    for i in range( m ):
        # Reshape the array to 20x20
        image_array = X[i].reshape((20,20)).T

        # Normalize the array to the range 0-255
        normalized_array = 255 * (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

        # Convert to uint8
        image_array = normalized_array.astype(np.uint8)

        # Create an image from the array
        image = ImageGrab.fromarray(image_array, mode='L')

        # Save the image
        image.save(f'PngData/{i}_{y[i,0]}.png')

    print(f'task done')
