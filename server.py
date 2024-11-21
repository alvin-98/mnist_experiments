from flask import Flask, render_template, jsonify
import json
import os
import torch
from torchvision import datasets, transforms
import random
import base64
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_logs')
def get_logs():
    try:
        with open('static/training_logs.json', 'r') as f:
            logs = json.load(f)
        return jsonify(logs)
    except FileNotFoundError:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True) 