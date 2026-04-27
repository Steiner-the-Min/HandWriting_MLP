import base64
import io
import keras
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

graph = tf.get_default_graph()
app = Flask(__name__)

json_path = os.path.join(BASE_DIR, 'model.json')
with open(json_path, 'r') as f:
    loaded_model_json = f.read()
model = model_from_json(loaded_model_json)

weights_path = os.path.join(BASE_DIR, 'model_weights.h5')
model.load_weights(weights_path)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print("模型已通过分离方式成功加载！")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(img_data)
    
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    
    img_array = np.array(image)
    img_array = 255 - img_array 
    
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 784).astype('float32')
    
    global graph
    with graph.as_default():
        predictions = model.predict(img_array)[0]
    image.save("test.png")
    results = [{"digit": i, "probability": float(predictions[i])} for i in range(10)]
    results = sorted(results, key=lambda x: x['probability'], reverse=True)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)