<!--
# FoodScan

FoodScan is a web application built using machine learning models to identify food items, retrieve recipes and compare nutrients. The application is developed using Flask and React.
-->
## Models

### 1. FoodModel
- **Dataset:** [Food-101](https://www.kaggle.com/datasets/kmader/food41)
- **Accuracy:** 78.18%

### 2. IndianFoodModel
- **Dataset:** [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)
- **Accuracy:** 75.26%

## Features
- **Food Detection:** Identify food items from images.
- **Recipe Retrieval:** Get recipes for identified food items.
- **Nutrient Comparison:** Compare the nutritional values of food items.

## Backend Setup
The backend is built using Flask API.

### Steps to Set Up:
1. Create a virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```

## Frontend Setup
The frontend is built using React.

### Steps to Set Up:
1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

## Website
Visit the live application here: [FoodScan](https://food-scan.vercel.app/)

## Screenshots

### Homepage  
  <img src="https://github.com/user-attachments/assets/ade71572-fdf9-46e3-935a-883374167489" alt="Homepage" width="75%">

### Features  
  <img src="https://github.com/user-attachments/assets/2bcf0bd6-4d0c-40f9-aa36-77bb9a004601" alt="Features" width="75%">

#### Food Detection  
  <img src="https://github.com/user-attachments/assets/208ca952-0975-405f-85f0-b1bc00db740e" alt="Food Detection" width="75%">

#### Recipe Retrieval  
 <img src="https://github.com/user-attachments/assets/110f79c1-28ef-4f91-8fef-746c4dbf6b77" alt="Nutrient Comparison 1" width="75%">

#### Nutrient Comparison  
<p >
  <img src="https://github.com/user-attachments/assets/b821c050-5305-40ff-848d-3b68e3be1b06" alt="Nutrient Comparison 1" width="45%">
  <img src="https://github.com/user-attachments/assets/d0d3c5d3-a347-475d-a722-3a0fcaa712cb" alt="Nutrient Comparison 2" width="45%">
</p>  

#### Diet Plan  
<p >
  <img src="https://github.com/user-attachments/assets/af075840-d763-49e1-ab49-c854cb508e0e" alt="Diet Plan 1" width="45%">
  <img src="https://github.com/user-attachments/assets/f570bd02-7d10-460e-bcf5-ba7da7c44776" alt="Diet Plan 2" width="45%">
</p>



