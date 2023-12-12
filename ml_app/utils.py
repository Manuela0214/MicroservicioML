import os
import pickle
from uuid import uuid4
from django.http import HttpResponseServerError, JsonResponse
import joblib
from pymongo import MongoClient
import pandas as pd  
from bson import ObjectId
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import is_classifier


from ml_app.database import get_database_connection

def load_dataset(dataset_id):
    try:
        client, db = get_database_connection()

        collection_name = "Datasets"
        collection = db[collection_name]

        dataset_document = collection.find_one({'_id': str(dataset_id)})

        if not dataset_document:
            raise ValueError(f"Dataset con ID {dataset_id} no encontrado en la colección {collection_name}")

        return dataset_document

    except Exception as e:
        raise e


def load_datasetImputation(dataset_id):
    try:
        client, db = get_database_connection()

        collection_name = "DatasetsImputacion"
        collection = db[collection_name]

        dataset_document = collection.find_one({'original_dataset_id': str(dataset_id)})

        if not dataset_document:
            raise ValueError(f"Dataset con ID {dataset_id} no encontrado en la colección {collection_name}")

        return dataset_document

    except Exception as e:
        raise e


def store_imputed_dataset(imputed_data, dataset_id):
    try:
        client, db = get_database_connection()
        new_collection = db["DatasetsImputacion"]

        new_dataset_id = str(uuid4()) 

        dataset_document = {
            '_id': new_dataset_id,
            'original_dataset_id': dataset_id,
            'data': imputed_data.to_dict(orient='records')
        }

        new_collection.insert_one(dataset_document)
        return {'mensaje': 'Datos almacenados con exito en la nueva coleccion.'}
    
    except Exception as e:
        return {'error': str(e)}


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Tipo no serializable: {type(obj)}")


def load_best_model(train_id):
    try:
        # Conectarse a la base de datos y obtener la información del mejor modelo con el ID proporcionado
        client, db = get_database_connection()
        collection_name = "TrainedModels"
        collection = db[collection_name]
        best_model_info = collection.find_one({'_id': train_id})

        if best_model_info is None:
            raise ValueError('Modelo entrenado no encontrado')

        # Imprime o registra toda la información del modelo para inspeccionar la estructura
        print(f"Información completa del modelo: {best_model_info}")

        # Asegúrate de utilizar la clave correcta para recuperar el modelo
        if 'model_data' not in best_model_info:
            raise ValueError("La clave 'model_data' no está presente en la información del mejor modelo")

        # Obtener el modelo desde el campo correspondiente en la base de datos
        #loaded_model = best_model_info['model_data']  # Ajusta esto según cómo esté almacenado tu modelo
        model_filename = best_model_info.get('model_filename')
        with open(model_filename, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        # Devolver el modelo cargado
        return loaded_model

    except Exception as e:
        raise ValueError(f'Error al cargar el modelo: {str(e)}')
