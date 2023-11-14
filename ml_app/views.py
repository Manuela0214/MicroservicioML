# ml_app/views.py
import json
from uuid import uuid4
import pandas as pd
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
from ml_app.database import get_database_connection
from ml_app.utils import load_dataset

def home(request):
    return JsonResponse({'mensaje': 'Bienvenido a la API de ML!'})

@csrf_exempt
def load(request):
    if request.method == 'POST' and request.FILES.get('archivo'):
        archivo = request.FILES['archivo']
        extension = archivo.name.split('.')[-1].lower()
        
        if extension in ['xls', 'xlsx', 'csv']:
            if extension in ['xls', 'xlsx']:
                df = pd.read_excel(archivo)
            elif extension == 'csv':
                try:
                    # df = pd.read_csv(pd.compat.StringIO(csv_data), sep=';')
                    df = pd.read_csv(archivo)
                except Exception as e:
                    return JsonResponse({'error': f'Error al leer el archivo CSV: {str(e)}'})

            print(df.head())
            json_data = df.to_dict(orient='records')
            
            #client = MongoClient("mongodb+srv://user_mongo_ml:admin123@clustermongo.su7wh1s.mongodb.net/?retryWrites=true&w=majority")
            #db = client["BDInteligentes"]
            client, db = get_database_connection()
            dataset_id = str(uuid4())
            #collection = db["coleccionML"]
            collection = db[f"coleccionML_{dataset_id}"]
            #collection.delete_many({})
            collection.insert_many(json_data)
            print(json_data)
            
            return JsonResponse({'mensaje': f'Archivo {extension} cargado y almacenado en MongoDB con exito'})
        else:
            return JsonResponse({'error': 'El archivo debe ser un documento Excel o CSV'})
    else:
        return JsonResponse({'error': 'Solicitud no valida. Debe ser una solicitud POST'})
    
@csrf_exempt
def basic_statistics(request, dataset_id):
    try:
        if request.method == 'GET':
            dataset = load_dataset(dataset_id)
            statistics = dataset.describe()
            result = statistics.to_dict()

            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Solicitud no valida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    
@csrf_exempt
def columns_describe(request, dataset_id):
    try:       
        if request.method == 'GET':
            dataset = load_dataset(dataset_id)
            column_types = dataset.dtypes.apply(lambda x: 'Texto' if 'object' in str(x) else 'Numérico').to_dict()
            column_types.pop('_id')
            return JsonResponse(column_types)
        else:
            return JsonResponse({'error': 'Solicitud no valida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})