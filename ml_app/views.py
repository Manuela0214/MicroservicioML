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
        archivo = request.FILES['archivo']
        extension = archivo.name.split('.')[-1].lower()
        
        if extension in ['xls', 'xlsx', 'csv']:
            if extension in ['xls', 'xlsx']:
                # Cargar el archivo Excel
                df = pd.read_excel(archivo)
            elif extension == 'csv':
                # Cargar el archivo CSV
                df = pd.read_csv(archivo)

            # Convertir el DataFrame de Pandas a un diccionario JSON
            json_data = df.to_dict(orient='records')
            
            # Almacena el JSON en MongoDB
            client = MongoClient("mongodb+srv://user_mongo_ml:admin123@clustermongo.su7wh1s.mongodb.net/?retryWrites=true&w=majority")
            db = client["BDInteligentes"]
            collection = db["coleccionML"]
            collection.insert_many(json_data)
            print(json_data)
            
            return JsonResponse({'mensaje': f'Archivo {extension} cargado y almacenado en MongoDB con éxito'})
        else:
            return JsonResponse({'error': 'El archivo debe ser un documento Excel o CSV'})
    else:
        return JsonResponse({'error': 'Solicitud no válida'})