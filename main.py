from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class Iris_type(BaseModel):
    
    sepal_length: int # 1, // Integer
    sepal_width : int # 5, // Integer
    petal_length: int # 5, // Integer
    petal_width : int # 3  // Integer

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)   



@app.post('/')
async def scoring_endpoint(item: Iris_type):
    df = pd.DataFrame([item.dict().values()], columns = item.dict().keys())
    y_pred = model.predict(df)
    return {"predictions" : int(y_pred)}