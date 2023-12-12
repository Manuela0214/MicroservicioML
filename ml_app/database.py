import json
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from decouple import config  # Cambiado desde os.getenv
load_dotenv()

def get_database_connection():
    
    #with open('config.json', 'r') as file:
     #   config = json.load(file)

    mongo_username = config('DB_USER')
    mongo_password = config('DB_PASSWORD')
    mongo_cluster = config('DB_CLUSTER')
    mongo_database = config('DB_NAME')
    mongo_uri= f"mongodb+srv://{mongo_username}:{mongo_password}@{mongo_cluster}.gmyir2o.mongodb.net/?retryWrites=true&w=majority"
    #mongo_uri = f"mongodb://localhost:27017"

    client = MongoClient(mongo_uri)
    db = client[mongo_database]

    return client, db
