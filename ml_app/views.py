# ml_app/views.py
import json
import pandas as pd
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt

def home(request):
    return JsonResponse({'mensaje': 'Bienvenido a la API de ML!'})

@csrf_exempt
def cargar_excel(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        excel_file = request.FILES['archivo']
        if excel_file.name.endswith('.xls') or excel_file.name.endswith('.xlsx'):
            # Cargar el archivo Excel y convertirlo a JSON
            df = pd.read_excel(excel_file)
            #json_data = json.loads(df.to_json(orient='records'))
            json_data = df.to_dict(orient='records')
            # Almacena el JSON en MongoDB
            client = MongoClient("mongodb+srv://<user_mongo_ml>:<rhLlLiR5fcJd3Mjb>@clustermongo.6vnczd1.mongodb.net/?retryWrites=true&w=majority")
            db = client["DB_ML_inteligentes"]
            collection = db["coleccion"]
            collection.insert_many(json_data)
            
            return JsonResponse({'mensaje': 'Archivo Excel cargado y almacenado en MongoDB con éxito'})
        else:
            return JsonResponse({'error': 'El archivo debe ser un documento Excel'})
    else:
        return JsonResponse({'error': 'Solicitud no válida'})
