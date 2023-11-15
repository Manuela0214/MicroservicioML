from uuid import uuid4
from pymongo import MongoClient
import pandas as pd  
from bson import ObjectId

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

