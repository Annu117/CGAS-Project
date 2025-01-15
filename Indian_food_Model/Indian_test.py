import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


image_path = r"C:\Users\Dell\OneDrive\Desktop\Indian Food\Indian Food Images\chapati\9b128b45f7.jpg"
images_df = pd.read_csv('indian_data.csv')
Image_Classes = list(images_df['Label'].unique())

# Load the trained model
print("Loading model...")
model = load_model('indian_best.keras')

def predict_and_show(image_path, model, show=False):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
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









# def predict_images(filenames , model):
#     cols = 5
#     images = []
#     total_images = len(os.listdir(filenames))
#     rows = total_images // cols + 1

#     # fig = plt.figure(1 , figsize = (50,50))
#     fig = plt.figure(figsize=(cols * 8, rows * 8))  # Adjusted for larger images
#     # fig = plt.figure(figsize=(cols * 6, rows * 6)) 
    
#     for i in sorted(os.listdir(filenames)):
#         images.append(os.path.join(filenames , i))
        
#     for subplot , imgs in enumerate(images):
#         img_ = image.load_img(imgs , target_size = (299 , 299))
#         img_array = image.img_to_array(img_)
#         img_processed = np.expand_dims(img_array , axis = 0)
#         img_processed /= 255.
        
#         prediction = model.predict(img_processed)
#         index = np.argmax(prediction)
        
#         preds = Image_Classes[index]
        
            
#         fig = plt.subplot(rows , cols , subplot+1)
#         fig.set_title(preds , pad = 5 , size = 8)
#         plt.imshow(img_array)
    
#     # Reduce the space between rows (vertical spacing)
#     plt.subplots_adjust(hspace=0.005)  # Adjust vertical spacing between rows
#     plt.tight_layout()
#     plt.show()  
    
# predict_images(image_folder_path,model)



# def predict_images(filenames, model):
#     cols = 5  # Number of columns
#     images = []
#     total_images = len(os.listdir(filenames))  # Total images in the folder
#     rows = total_images // cols + 1  # Calculate rows dynamically

#     # Increase the figure size to display larger images
#     fig = plt.figure(figsize=(cols * 8, rows * 8))  # Adjusted for larger images

#     for i in sorted(os.listdir(filenames)):
#         images.append(os.path.join(filenames, i))

#     for subplot, imgs in enumerate(images):
#         img_ = image.load_img(imgs, target_size=(299, 299))  # Resize image to model input size
#         img_array = image.img_to_array(img_)
#         img_processed = np.expand_dims(img_array, axis=0)
#         img_processed /= 255.  # Normalize pixel values

#         prediction = model.predict(img_processed)  # Get prediction
#         index = np.argmax(prediction)
#         preds = Image_Classes[index]

#         # Adjust subplot to reduce space and increase image size
#         ax = plt.subplot(rows, cols, subplot + 1)  # Create subplot
#         ax.set_title(preds, pad=5, size=10)  # Title font size
#         plt.imshow(img_array)  # Display image
#         plt.axis('off')  # Hide axis for clarity

#     # Reduce the space between rows (vertical spacing) and columns (horizontal spacing)
#     plt.subplots_adjust(hspace=0.2, wspace=0.1)  # Adjust horizontal (wspace) and vertical (hspace) spacing
#     plt.tight_layout()  # Ensure tight layout
#     plt.show()  # Display all images with predictions

# predict_images(image_folder_path, model)
