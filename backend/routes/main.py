import cv2
import numpy as np

from fastapi import APIRouter, Form, UploadFile, File

from blood_report_analyzer.main import analyze_blood_sugar_report
from diet_plan_recommender.main import get_meal_plan
from nutrition_need_calculator.diseases import get_diseases
from nutrition_need_calculator.need_calculator import get_dietary_need
from medic.main import calculate_bmi, calculate_dream_weight
from workout_routine_recommender.recommender import predict_workout_plan

router = APIRouter(
    prefix="/api",
    tags=["core"],
    responses={404: {"description": "The requested URL was not found"}},
)


@router.post("/")
async def root(
        height: int = Form(...),
        weight: int = Form(...),
        age: int = Form(...),
        gender: str = Form(...),
        image: UploadFile = File(...),
        diseases_info: str = Form(None),
):
    # Validate gender
    if gender.lower() not in ["male", "female"]:
        return {"error": "Invalid gender. Please provide 'male' or 'female'."}

    bmi = calculate_bmi(weight, height)
    dream_weight = calculate_dream_weight(weight, bmi)
    if image and image.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Only jpeg and png images are supported."}

    diseases = None
    try:
        if image:
            contents = await image.read()
            if not contents:    # Check if the image file is not empty
                return {"error": "The image is empty."}
            nparray = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

            # Analyze blood sugar levels from the image
            # blood_sugar_level = analyze_blood_sugar_report(img)
            # diseases = get_diseases(blood_sugar_level, bmi)
            diseases = get_diseases(None, bmi)
        else:
            diseases = get_diseases(None, bmi)
    except Exception as e:
        return {"error": f"Error processing image or analyzing blood report: {str(e)}"}

    # Calculate nutritional needs
    nutrition_need = get_dietary_need(weight, height, age, gender.lower())
    
    # Generate workout plan
    workout_plan = predict_workout_plan(gender, age, weight, dream_weight, bmi)

    # Generate meal plan
    meal_plan = get_meal_plan(['low_sodium_diet', 'low_fat_diet'], diseases, ['calcium', 'vitamin_c'], ['non-veg'], 'i love indian')


    # Include doctor's validation if diseases_info is provided
    if diseases_info:
        workout_plan = workout_plan

    return {
        "need": nutrition_need,
        "workout_plan": workout_plan,
        "meal_plan": meal_plan,
    }
