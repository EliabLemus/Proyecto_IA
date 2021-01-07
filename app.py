import csv
import pickle, os, glob
from io import StringIO
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
import NeuralNetwork
hyper = []
# Modelo entrenado
# with open('TrainedModels/usac_model.dat', 'rb') as f:
#     models = pickle.load(f)
#     usac_model = models[0]
    
app = Flask(__name__)
app.config['FLASK_DEBUG'] = True
app.debug=1
# @app.after_request
# def add_header(r):
#     """
#     Add headers to both force latest IE rendering engine or Chrome Frame,
#     and also to cache the rendered page for 10 minutes.
#     """
#     r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
#     r.headers["Pragma"] = "no-cache"
#     r.headers["Expires"] = "0"
#     r.headers['Cache-Control'] = 'public, max-age=0'
@app.route("/", methods=['GET', 'POST'])
def hello():
    if request.method == 'GET':
        return render_template('form.html')
    if request.method == 'POST':
        print('inside post')
        #AQUI MOSTRAMOS LO DEL MODELO
        with open('TrainedModels/best_model.dat', 'rb') as f:
            best_model = pickle.load(f)
            NeuralNetwork.show_Model([best_model])
        return render_template('form.html')
    
@app.route("/hyper", methods=['GET'])
def hyper():
    if request.method == 'GET':
        with open('TrainedModels/HyperParameters.dat', 'rb') as f:
            hyper_data = pickle.load(f)
        return render_template('hyper.html', hyper_data=hyper_data)