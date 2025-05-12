import joblib
from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import numpy as np

model: RandomForestClassifier = joblib.load('battery_performance_model.joblib')

app = FastAPI()
class Features(BaseModel):
    cell : float
    module: float
    pack : float
    energy_density :float
    power_density: float
    c_rate : float
    cycle_life : float
    efficiency : float
    self_discharge_rate : float
    thickness : float
    calendering_pressure : float
    min_temp :float
    max_temp : float

#features : list[float] = [15.5,346.55,2639.2,257.96,382.48,1.91,2530,97.38,1.03,76.43,123.02,-25.0,65.0]
#prediction = model.predict([features])
#print(prediction)

@app.post("/batteryperfromance/")
def predict(features: Features):
    feature_list = [
        features.cell,
        features.module,
        features.pack,
        features.energy_density,
        features.power_density,
        features.c_rate,
        features.cycle_life,
        features.efficiency,
        features.self_discharge_rate,
        features.thickness,
        features.calendering_pressure,
        features.min_temp,
        features.max_temp,
    ]
    
    input_data = np.array(feature_list).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"batterygrade": prediction[0]}