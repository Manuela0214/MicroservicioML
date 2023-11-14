# ml_app/utils.py
from pymongo import MongoClient
import pandas as pd  # Agrega esta línea para importar pandas
from bson import ObjectId

from ml_app.database import get_database_connection

def load_dataset(dataset_id):
    try:
        client, db = get_database_connection()

        collection_name = dataset_id
        collection = db[collection_name]

        cursor = collection.find()

        document_count = collection.count_documents({})
        
        if document_count > 0:
            data_list = list(cursor)

            df = pd.DataFrame(data_list)

            df['_id'] = df['_id'].apply(lambda x: str(x) if isinstance(x, ObjectId) else x)
        else:
            raise ValueError(f"Dataset con ID {dataset_id} no encontrado en la colección {collection_name}")

        return df

    except Exception as e:
        raise e
