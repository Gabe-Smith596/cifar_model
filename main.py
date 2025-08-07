#Importing Libraries
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model


#Loading the Model
model = load_model("cifar_model.keras")

#Initialisingt the Fast API
app = FastAPI()

# Creating the Class Names so that we are getting Predictions in Words rather than numbers
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


#Creating Logic for Handling Input Images
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB') #Opening and Converting to RGB
    image = image.resize((32,32)) # Resizing the Image
    image_array = np.array(image)/255.0 #Normalising the Image Values to a range of 0-1
    image_array = np.expand_dims(image_array, axis=0) #Addingt the batch dimension (RGB or Greyscale value) to create 32x32x3
    return image_array


#First API Route
@app.post("/predict/")
async def predict(file: UploadFile = File()):
    contents = await file.read() #Read the file contents
    image_array = preprocess_image(contents) #Calls the preprocess function
    predictions = model.predict(image_array) #Make the predictions
    predicted_class = class_names[np.argmax(predictions[0])] #Get the class label with the highest probability
    return JSONResponse(content={"class": predicted_class})

@app.get("/")
async def root():
    return {"message": "Welcome to the Cifar-10 image classification API. Upload an image get predictions at /predict/."}