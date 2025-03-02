from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load the trained model from a pickle file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Route to render the Iris form
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to handle form submission and prediction
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sepal_length: float = Form(...), sepal_width: float = Form(...),
                  petal_length: float = Form(...),
                  petal_width: float = Form(...)):
    
    # Make a prediction using the model
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_class = prediction[0]
    return templates.TemplateResponse("index.html", { "request": request, "result": f"Predicted Iris Class: {predicted_class}" })