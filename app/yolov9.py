import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from roboflow import Roboflow
# import os
# import tempfile
from PIL import Image

rf = Roboflow(api_key="qeurMItsrfyR1IsxYkDb")
project = rf.workspace("reduceddataset").project("reduce-zub6u")
version = project.version(8)
model = version.model

app = FastAPI()


def resize_image(image, target_size=(416, 416)):
    """
    Resize the image to the target size.
    """
    resized_image = cv2.resize(image, target_size)
    return resized_image


def serialize_response(response):
    """
    Convert the prediction response to the desired format.
    """
    predictions = []
    for prediction in response:
        prediction_data = {
            'x': prediction['x'],
            'y': prediction['y'],
            'width': prediction['width'],
            'height': prediction['height'],
            'confidence': prediction['confidence'],
            'class': prediction['class'],
            'class_id': prediction['class_id'],
            'detection_id': prediction['detection_id'],
            # 'image_path': prediction['image_path'],
            # 'prediction_type': prediction['prediction_type']
        }
        predictions.append(prediction_data)

    return {'predictions': predictions}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Generate a unique temporary filename
    content = await image.read()
    pil_image = Image.open(io.BytesIO(content))
    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_BGR2RGB)
    target_size = (416, 416)  # Adjust the target size as needed
    resized_image = resize_image(cv2_image, target_size)
    # results = model(resized_image)
    try:
        response = model.predict(resized_image, confidence=40, overlap=30)
    except Exception as e:
        return {"error": f"Failed to get predictions from Roboflow: {str(e)}"}
    result = serialize_response(response)
    print(result)

    def handle_aloevera():
        return {"predictions": [{"leaf_name": "Aloevera", "confidence": round(float(predictions[0]["confidence"]), 2)}]}

    def handle_lemon():
        return {"predictions": [{"leaf_name": "Lemon", "confidence": round(float(predictions[0]["confidence"]), 2)}]}

    def handle_mint():
        return {"predictions": [{"leaf_name": "Mint", "confidence": round(float(predictions[0]["confidence"]), 2)}]}

    def handle_neem():
        return {"predictions": [{"leaf_name": "Neem", "confidence": round(float(predictions[0]["confidence"]), 2)}]}

    def handle_tulsi():
        return {"predictions": [{"leaf_name": "Tulsi", "confidence": round(float(predictions[0]["confidence"]), 2)}]}

    def handle_default():
        return {"predictions": [{"leaf_name": "Undefined class", "confidence": 0.0}]}

    predictions = result['predictions'] if result else None

    switch_cases = {
        'Aloevera': handle_aloevera,
        'Lemon': handle_lemon,
        'Mint': handle_mint,
        'Neem': handle_neem,
        'Tulsi': handle_tulsi,
        'default': handle_default
    }

    predicted_name = predictions[0]["class"] if predictions else None  # Assuming only one prediction is made
    handler = switch_cases.get(predicted_name, switch_cases['default'])
    return handler()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="localhost", port=8002)