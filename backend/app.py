import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import gradio as gr  # Gradio serves as an API backend

# Load class labels
images_df = pd.read_csv('data.csv')
Image_Classes = list(images_df['Label'].unique())

# Load the trained model
print("Loading model...")
# model = load_model('best_model.keras')
model = load_model('indian_best.keras')

def predict_image(img):
    # Preprocess the image
    img = img.resize((200, 200))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    pred = model.predict(img_array)
    index = np.argmax(pred)
    pred_value = Image_Classes[index]
    
    return {"class": pred_value, "confidence": float(np.max(pred))}

# Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    img = image.load_img(file, target_size=(200, 200))
    result = predict_image(img)
    return jsonify(result)

# Optional: Add Gradio for local testing
def gradio_interface():
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="pil"),  # 'pil' type will pass a PIL image
        outputs="json"  # Return a JSON object with the predicted class and confidence
    )
    interface.launch(share=True)

if __name__ == '__main__':
    # gradio_interface()
    app.run(host="0.0.0.0", port=5000)
