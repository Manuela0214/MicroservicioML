import json
from pymongo import MongoClient

def get_database_connection():
    with open('config.json', 'r') as file:
        config = json.load(file)

    mongo_username = config['mongo']['username']
    mongo_password = config['mongo']['password']
    mongo_host = config['mongo']['host']
    mongo_database = config['mongo']['database']

    mongo_uri = f"mongodb://localhost:27017"

    client = MongoClient(mongo_uri)
    db = client[mongo_database]

    return client, db
