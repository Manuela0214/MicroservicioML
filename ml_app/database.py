# ml_app/database.py
import json
from pymongo import MongoClient

def get_database_connection():
    # Lee la configuración desde el archivo JSON
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Obtiene los valores necesarios para la conexión a MongoDB
    mongo_username = config['mongo']['username']
    mongo_password = config['mongo']['password']
    mongo_host = config['mongo']['host']
    mongo_database = config['mongo']['database']

    # Construye la cadena de conexión a MongoDB
    mongo_uri = f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_host}/{mongo_database}?retryWrites=true&w=majority"

    # Establece la conexión a MongoDB y devuelve el cliente y la base de datos
    client = MongoClient(mongo_uri)
    db = client[mongo_database]

    return client, db
