import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K 
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def get_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    global classifier
    classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model.h5")
    #la puta hostia
    classifier._make_predict_function()
    print(" * Model loaded!")

def preprocess_image(image_select, target_size):
    if image_select.mode != "RGB":
        image_select = image_select.convert("RGB")    
    image_select = image_select.resize(target_size)
    image_select = img_to_array(image_select)
    image_select = np.expand_dims(image_select, axis = 0)
    return image_select

print("* Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force= True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image_select = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image_select,target_size=(64,64))
    result = classifier.predict(processed_image)
    if result[0][0] >= 0.5:
        prediction = 'Nudge'
    elif result[0][1] >= 0.5:
        prediction = 'Persuasive'
    else:
        prediction = 'Unpleasant' 
    response = {
        'prediction': {
            'image': prediction
        }
    }
    return jsonify(response)
