''''
'from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import tensorflow as tf

# Load saved model, scaler, and encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

# Define the input format
class InputData(BaseModel):
    Ship_Mode: str
    Segment: str
    City: str
    State: str
    Region: str
    Category: str
    Sub_Category: str
    Order_Date: str  # Format: YYYY-MM-DD
    Ship_Date: str   # Format: YYYY-MM-DD

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()

    # Calculate shipping time
    try:
        order_date = datetime.strptime(input_dict["Order_Date"], "%Y-%m-%d")
        ship_date = datetime.strptime(input_dict["Ship_Date"], "%Y-%m-%d")
        shipping_time = (ship_date - order_date).days
        if shipping_time < 0:
            return {"error": "Ship date cannot be before order date."}
    except Exception as e:
        return {"error": f"Date parsing error: {e}"}

    input_dict["shipping_time"] = shipping_time

    # Encode categorical features
    for col in encoders:
        try:
            input_dict[col] = encoders[col].transform([input_dict[col]])[0]
        except ValueError:
            return {"error": f"Invalid value for column '{col}': {input_dict[col]}"}

    # Define feature order
    feature_order = ['Ship_Mode', 'Segment', 'City', 'State', 'Region', 'Category', 'Sub_Category', 'shipping_time']

    # Create input array
    input_array = np.array([[input_dict[col] for col in feature_order]])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)
    predicted_sales = float(prediction[0][0])

    return {"predicted_sales": predicted_sales}
'''

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import tensorflow as tf

# Load saved Keras model, scaler, and encoders
model = tf.keras.models.load_model('model.h5', compile=False)  # updated line
 # updated to load .h5 model
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

# Define the input format
class InputData(BaseModel):
    Ship_Mode: str
    Segment: str
    City: str
    State: str
    Region: str
    Category: str
    Sub_Category: str
    Order_Date: str  # Format: YYYY-MM-DD
    Ship_Date: str   # Format: YYYY-MM-DD

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()

    # Calculate shipping time
    try:
        order_date = datetime.strptime(input_dict["Order_Date"], "%Y-%m-%d")
        ship_date = datetime.strptime(input_dict["Ship_Date"], "%Y-%m-%d")
        shipping_time = (ship_date - order_date).days
        if shipping_time < 0:
            return {"error": "Ship date cannot be before order date."}
    except Exception as e:
        return {"error": f"Date parsing error: {e}"}

    input_dict["shipping_time"] = shipping_time

    # Encode categorical features
    for col in encoders:
        try:
            input_dict[col] = encoders[col].transform([input_dict[col]])[0]
        except ValueError:
            return {"error": f"Invalid value for column '{col}': {input_dict[col]}"}

    # Define feature order
    feature_order = ['Ship_Mode', 'Segment', 'City', 'State', 'Region', 'Category', 'Sub_Category', 'shipping_time']

    # Create input array
    input_array = np.array([[input_dict[col] for col in feature_order]])

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Predict using the Keras model
    prediction = model.predict(input_scaled)
    predicted_sales = float(prediction[0][0])  # assuming the output is a single value

    return {"predicted_sales": predicted_sales}
