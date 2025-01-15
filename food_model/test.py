import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


image_path = r"C:\Users\Dell\OneDrive\Desktop\image.png"
images_df = pd.read_csv('data.csv')
Image_Classes = list(images_df['Label'].unique())
Image_Classes.sort()

# Load the trained model
print("Loading model...")
model = load_model('best_model.h5')

def predict_and_show(image_path, model, show=False):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize image
    
    # Make prediction
    pred = model.predict(img_array)
    index = np.argmax(pred)
    pred_value = Image_Classes[index]
    
    if show:
        plt.imshow(img)  # Original image (not array) for better visualization
        plt.axis('off')
        plt.title(f"Predicted: {pred_value}")
        plt.show()
    
    return pred_value, np.max(pred)

# Predict and display results
predicted_class, confidence = predict_and_show(image_path, model, show=True)  # Set show=True to display the image
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}")


