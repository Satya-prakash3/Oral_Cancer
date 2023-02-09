from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model,Model
from keras.preprocessing import image
from keras.utils import load_img,img_to_array
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from util import base64_to_pil
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# Load your trained model
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        img = load_img(file_path , target_size=(224,224))
        img= img_to_array(img)
       # img=np.true_divide(img,255)
        img=np.expand_dims(img,axis=0)
        model = load_model('Saved_Models/model.h5')
        result=model.predict(img)
        result=np.round(result)
        if result==0:
            result="Normal"
        elif result==1:
            result="OSCC"
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')