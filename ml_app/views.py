import json
from uuid import uuid4
from bson import ObjectId
import pandas as pd
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
from ml_app.database import get_database_connection
from ml_app.utils import load_dataset, store_imputed_dataset

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
                    df = pd.read_csv(archivo)
                except Exception as e:
                    return JsonResponse({'error': f'Error al leer el archivo CSV: {str(e)}'})

            client, db = get_database_connection()
            collection_name = "Datasets"
            dataset_document = {
                '_id': str(uuid4()), 
                'data': df.to_dict(orient='records')  
            }
            collection = db[collection_name]
            collection.insert_one(dataset_document)
            
            return JsonResponse({'mensaje': f'Archivo {extension} cargado y almacenado en MongoDB con éxito. Dataset ID: {dataset_document["_id"]}'})
        else:
            return JsonResponse({'error': 'El archivo debe ser un documento Excel o CSV'})
    else:
        return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'})
 
@csrf_exempt
def basic_statistics(request, dataset_id):
    try:
        if request.method == 'GET':
            dataset = load_dataset(dataset_id)            
            dataset_data = dataset.get('data', [])
            
            if not dataset_data:
                return JsonResponse({'error': 'No se encontraron datos en el dataset'}, status=404)

            df = pd.DataFrame(dataset_data)

            statistics = df.describe()
            result = statistics.to_dict()

            return JsonResponse(result)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    
@csrf_exempt
def columns_describe(request, dataset_id):
    try:
        if request.method == 'GET':
            dataset = load_dataset(dataset_id)
            dataset_data = dataset.get('data', [])
            column_types = pd.DataFrame(dataset_data).dtypes.apply(lambda x: 'Texto' if 'object' in str(x) else 'Numérico').to_dict()
            column_types.pop('_id', None)

            return JsonResponse(column_types)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})


@csrf_exempt
def imputation(request, dataset_id, number_type):
    try:
        if request.method == 'POST':
            original_dataset = load_dataset(dataset_id)
            dataset = pd.DataFrame(original_dataset.get('data', [])).copy()
            if number_type == '1':
                dataset = dataset.dropna()
            elif number_type == '2':
                for column in dataset.columns:
                    if dataset[column].dtype == 'object':
                        dataset[column].fillna(dataset[column].mode()[0], inplace=True)
                    else:
                        dataset[column].fillna(dataset[column].mean(), inplace=True)
            else:
                return JsonResponse({'error': 'Tipo de imputación no válido. Utilice 1 o 2.'}, status=400)

            store_imputed_dataset(dataset, dataset_id)

            return JsonResponse({'mensaje': 'Imputación realizada con éxito', 'new_dataset': dataset.to_dict(orient='records')})

        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'})

    except Exception as e:
        return JsonResponse({'error': str(e)})
