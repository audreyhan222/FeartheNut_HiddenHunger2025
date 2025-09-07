from flask import render_template, request, redirect, url_for, flash, session
import re
import random
from .model import predict_risk
from openai import OpenAI
from dotenv import load_dotenv
import os
import google.generativeai as genai

def calculate_vitamin_intake(food_description, calories, age, gender):
    """
    AI-powered vitamin calculation based on food description and calories.
    This simulates an AI analysis of the food description to estimate vitamin content.
    """
    
    # Convert description to lowercase for analysis
    load_dotenv()
    food_description = food_description.lower()
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""You are a vitamin calculator. Analyze the diet provided and calculate the daily intake of:
        - Vitamin A (μg)
        - Vitamin D (IU) 
        - Zinc (mg)
        - Iron (mg)
        - Folate (μg)
        
        Daily diet: {food_description}
        Caloric Intake: {calories} calories
        
        Output ONLY a single line in this exact format: vitamin_a,vitamin_d,zinc,iron,folate
        No extra text, no quotes, no explanations. All values should be numbers."""
        
        response = model.generate_content(prompt)
    except Exception as e:
        print(f"Could not calculate.")

    vitamin_list = response.text.split(",")
    return vitamin_list
    # features_list = [age, gender, income, education, vitamin_results['vitamin_a'], vitamin_results['vitamin_d'], vitamin_results['zinc'], vitamin_results['iron'], vitamin_results['folate']]
    # risk_level_int = predict_risk(features_list)
    


    












    # # Initialize base values
    # vitamin_a = 0
    # vitamin_d = 0
    # zinc = 0
    # iron = 0
    # folate = 0
    
    # # Food-based vitamin estimation (simplified AI logic)
    # # Vitamin A sources
    # if any(word in desc_lower for word in ['carrot', 'sweet potato', 'spinach', 'kale', 'broccoli', 'mango', 'apricot', 'pumpkin']):
    #     vitamin_a += random.uniform(200, 800)  # μg
    
    # # Vitamin D sources
    # if any(word in desc_lower for word in ['salmon', 'tuna', 'mackerel', 'sardine', 'egg', 'milk', 'cheese', 'fortified']):
    #     vitamin_d += random.uniform(5, 25)  # μg
    
    # # Zinc sources
    # if any(word in desc_lower for word in ['meat', 'beef', 'pork', 'chicken', 'oyster', 'crab', 'lobster', 'nuts', 'seeds']):
    #     zinc += random.uniform(5, 15)  # mg
    
    # # Iron sources
    # if any(word in desc_lower for word in ['red meat', 'beef', 'liver', 'spinach', 'beans', 'lentils', 'quinoa', 'dark chocolate']):
    #     iron += random.uniform(8, 20)  # mg
    
    # # Folate sources
    # if any(word in desc_lower for word in ['leafy greens', 'spinach', 'lettuce', 'broccoli', 'asparagus', 'beans', 'lentils', 'avocado']):
    #     folate += random.uniform(100, 400)  # μg
    
    # # Calorie-based adjustments
    # calorie_factor = calories / 2000  # Normalize to 2000 calorie diet
    
    # # Apply calorie scaling
    # vitamin_a *= calorie_factor
    # vitamin_d *= calorie_factor
    # zinc *= calorie_factor
    # iron *= calorie_factor
    # folate *= calorie_factor
    
    # # Age and gender adjustments
    # if age < 18:
    #     # Adolescents need more nutrients
    #     vitamin_a *= 1.2
    #     zinc *= 1.3
    #     iron *= 1.4
    #     folate *= 1.2
    # elif age > 65:
    #     # Elderly may have different absorption
    #     vitamin_d *= 1.5
    #     zinc *= 0.8
    
    # if gender.lower() == 'female':
    #     # Women typically need more iron
    #     iron *= 1.3
    #     folate *= 1.2
    
    # # Ensure minimum values based on typical diet
    # vitamin_a = max(vitamin_a, random.uniform(50, 200))
    # vitamin_d = max(vitamin_d, random.uniform(2, 10))
    # zinc = max(zinc, random.uniform(3, 8))
    # iron = max(iron, random.uniform(4, 12))
    # folate = max(folate, random.uniform(80, 300))
    
    # return {
    #     'vitamin_a': round(vitamin_a, 1),
    #     'vitamin_d': round(vitamin_d, 1),
    #     'zinc': round(zinc, 1),
    #     'iron': round(iron, 1),
    #     'folate': round(folate, 1)
    # }

def form():
    # Check if user is logged in
    if 'user_email' not in session:
        flash('Log in before you can access the functions', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        age = int(request.form.get('age', 0))
        gender = request.form.get('gender', '')
        income = request.form.get('income', '')
        education = request.form.get('education', '')
        state = request.form.get('state', '')
        food_description = request.form.get('food_description', '')
        calories = int(request.form.get('calories', 0))
        
        # Calculate vitamin intake using AI logic
        vitamin_results = calculate_vitamin_intake(food_description, calories, age, gender)
        
        # Put into our prediction model
        features_list = [age, gender, income, education, vitamin_results[0], vitamin_results[1], vitamin_results[2], vitamin_results[3], vitamin_results[4]]
        risk_level_int = predict_risk(features_list)

        # Calculate risk level
        risk_level = 'Low risk'
        if risk_level_int == 1:
            risk_level = 'High risk'

        vitamin_dictionary = {'vitamin_a': vitamin_results[0],
                              'vitamin_d': vitamin_results[1],
                              'zinc': vitamin_results[2], 
                              'iron': vitamin_results[3],
                              'folate': vitamin_results[4]
        }

        
        return render_template('vitamin_results.html', 
                             age=age,
                             gender=gender,
                             income=income,
                             education=education,
                             state=state,
                             food_description=food_description,
                             calories=calories,
                             vitamin_results=vitamin_dictionary,
                             risk_level=risk_level)
    
    return render_template('form.html')
