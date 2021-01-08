import csv
import pickle, os, glob
from io import StringIO
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_bootstrap import Bootstrap
import NeuralNetwork,GeneticAlgorithm
import Plotter
hyper = []
to_predict = dict()
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
        to_predict["Estado"]='Activo' 
        to_predict["Genero"]='FEMENINO'
        to_predict["edad"]='39'
        to_predict["cod_depto"]='1'
        to_predict["cod_muni"]='1'
        to_predict["Año"]='2015'
        return render_template('form.html', to_predict = to_predict)
    if request.method == 'POST':
        print('inside post')
        #AQUI MOSTRAMOS LO DEL MODELO
        with open('TrainedModels/best_model.dat', 'rb') as f:
            best_model = pickle.load(f)
            Plotter.show_Model([best_model])
            
            #para predecir: 
            #{'Estado': 'Traslado', 'Genero': 'MASCULINO', 'edad': '39', 'cod_depto': '1', 'nombre': 'Guatemala', 'cod_muni': '1', 'municipio': 'Ciudad de Guatemala', 'A\x96o': '2015'}
            
            
            gender = request.form['gender']
            age = request.form['age']
            year = request.form['year']
            cod_depto = request.form['department']
            cod_muni = request.form['municipio']
            to_predict["Estado"]='Activo'
            to_predict["Genero"]=gender
            to_predict["edad"]=age
            to_predict["cod_depto"]=cod_depto
            to_predict["cod_muni"]=cod_muni
            to_predict["Año"]=year
            print('to_predict:', to_predict)
            
            #distancia: 
            test = NeuralNetwork.initNeuralNetworkSingle(data=to_predict)

            result = best_model.predict(test)
            # result = 'No se cambiara' + 'distancia: ' + test 
            print('result from predict:', result)
        return render_template('form.html', result=result, to_predict = to_predict)
    
@app.route("/hyper", methods=['GET'])
def hyper():
    if request.method == 'GET':
        with open('TrainedModels/best_model.dat', 'rb') as f:
            best_model = pickle.load(f)
            Plotter.show_Model([best_model])
        with open('TrainedModels/HyperParameters.dat', 'rb') as f:
            hyper_data = pickle.load(f)
        return render_template('hyper.html', hyper_data=hyper_data,model=best_model)