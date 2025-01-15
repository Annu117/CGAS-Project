import './App.css';
import React, { useState, useEffect } from "react";
import FoodDetectionApp from './components/FoodDetectionApp';
import FoodIngredients from './components/RecipeRetrival';
import NutritionalDashboard from './components/NutritionalDashboard ';
import Home from './components/Home';
import Features from './components/Features';
import Header from './components/Header';
import Footer from './components/Footer'
import DietPlan from './components/diet/DietPlan';

function App() {
  return (
    <>
    <Header />
    <div id="home" className="min-h-screen mx-auto px-4 bg-blue-50">
      <Home />
    </div>
    <div id="features" className="min-h-screen mx-auto px-4 bg-purple-50">
      <Features />
    </div>
    <div id="Food-Detection" className="min-h-screen mx-auto px-4 bg-purple-50">
      <FoodDetectionApp />
    </div>
    <div id="Recipe-Retrival" className="min-h-screen mx-auto px-4 bg-purple-50">
      <FoodIngredients />
    </div>
    <div id="Nutrition-Comparision" className="min-h-screen mx-auto px-4 bg-purple-50">
      <NutritionalDashboard />
    </div>
    <div id="diet-plan" className="min-h-screen mx-auto px-4 bg-purple-50">
      <DietPlan />
    </div>
    <div id="diet" className="min-h-screen mx-auto px-4 bg-purple-50">
    </div>  
    <Footer/>
  
    </>
  );
}

export default App;
