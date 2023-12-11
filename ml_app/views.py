import json
from uuid import uuid4
from bson import ObjectId
from django.conf import settings
import pandas as pd
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
from ml_app.database import get_database_connection
from ml_app.utils import load_dataset, store_imputed_dataset,load_datasetImputation, convert_to_serializable
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import os


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

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

            numeric_statistics = df.describe()
            categorical_statistics = df.describe(include='O')

            numeric_result = numeric_statistics.to_dict()
            categorical_result = categorical_statistics.to_dict()

            numeric_result = json.loads(json.dumps(numeric_result, default=convert_to_serializable))
            categorical_result = json.loads(json.dumps(categorical_result, default=convert_to_serializable))
            result = {**numeric_result, **categorical_result}
            return JsonResponse(result, json_dumps_params={'indent': 2})
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


@csrf_exempt
def general_univariate_graphs(request, dataset_id):
    try:
        if request.method == 'POST':
            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])

            if dataset_data:
                df = pd.DataFrame(dataset_data)
                
                folder_path = os.path.join(settings.MEDIA_ROOT, 'univariate_graphs', dataset_id)

                os.makedirs(folder_path, exist_ok=True)
                
                images_paths = []
                for column in df.columns:
                    plt.figure(figsize=(8, 6))

                    if df[column].dtype == 'object':
                        sns.countplot(x=df[column])
                        plt.title(f'Count Plot for {column}')
                    else:
                        sns.kdeplot(df[column], cumulative=True)
                        plt.title(f'Probability Distribution for {column}')

                    image_path = os.path.join(folder_path, f'{column}_distribution.png')
                    plt.savefig(image_path)
                    plt.clf()
                    images_paths.append(image_path)

                return JsonResponse({'mensaje': 'Gráficos generados y almacenados con éxito.', 'images_paths': images_paths})
            else:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})

@csrf_exempt
def pca(request, dataset_id):
    try:
        if request.method == 'POST':
            client, db = get_database_connection()

            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])
            df = pd.DataFrame(dataset_data)

            potential_id_columns = ['No', 'Id', 'codigo','PassengerId']  
            id_column_name = next((col for col in potential_id_columns if col in df.columns), None)

            if id_column_name is None:
                df['_id'] = [str(uuid4()) for _ in range(len(df))]
                id_column_name = '_id'
               
            id_column = df[id_column_name]

            numerical_df = df.select_dtypes(include=[np.number])

            imputer = SimpleImputer(strategy='mean')
            numerical_df = pd.DataFrame(imputer.fit_transform(numerical_df), columns=numerical_df.columns)

            categorical_columns = df.select_dtypes(include=['object']).columns
            if len(categorical_columns) > 0:
                encoder = OneHotEncoder(drop='first', sparse=False)
                encoded_data = encoder.fit_transform(df[categorical_columns].fillna('Missing'))  # Imputar valores faltantes con una categoría 'Missing'
                numerical_df = pd.concat([numerical_df, pd.DataFrame(encoded_data)], axis=1)

            pca_model = PCA()
            transformed_data = pca_model.fit_transform(numerical_df)
            component_weights = pca_model.components_
            new_dataset_id = str(uuid4())
            new_dataset_document = {
                '_id': new_dataset_id,
                'original_dataset_id': dataset_id,
                'data': pd.DataFrame(transformed_data, columns=[f'component_{i}' for i in range(transformed_data.shape[1])]).to_dict(orient='records')
            }
            new_collection = db["DatasetsPCA"]
            new_collection.insert_one(new_dataset_document)

            return JsonResponse({
                'mensaje': 'PCA aplicado con éxito',
                'component_weights': component_weights.tolist(),
                'new_dataset_id': new_dataset_id
            })

        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'})

    except Exception as e:
        return JsonResponse({'error': str(e)})
    

@csrf_exempt
def bivariate_graphs_class(request, dataset_id):
    try:
        if request.method == 'GET':
            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])

            if dataset_data:
                df = pd.DataFrame(dataset_data)

                folder_path = os.path.join(settings.MEDIA_ROOT, 'bivariate_graphs', dataset_id)
                os.makedirs(folder_path, exist_ok=True)
                plt.figure(figsize=(12, 10))
                pair_plot = sns.pairplot(df)
                plot_path = os.path.join(folder_path, 'bivariate_graph.png')
                pair_plot.savefig(plot_path)
                plt.clf()

                return JsonResponse({'mensaje': 'Gráfico pair plot generado y almacenado con éxito.', 'plot_path': plot_path})
            else:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    

@csrf_exempt
def multivariate_graphs_class(request, dataset_id):
    try:
        if request.method == 'GET':
            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])

            if dataset_data:
                df = pd.DataFrame(dataset_data)

                folder_path = os.path.join(settings.MEDIA_ROOT, 'multivariate_graphs', dataset_id)
                os.makedirs(folder_path, exist_ok=True)
                correlation_matrix = df.corr()
                plt.figure(figsize=(12, 10))
                correlation_heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
                plot_path = os.path.join(folder_path, 'correlation_graph.png')
                correlation_heatmap.get_figure().savefig(plot_path)
                plt.clf()

                return JsonResponse({'mensaje': 'Gráfico de correlación generado y almacenado con éxito.', 'plot_path': plot_path})
            else:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud GET'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    

@csrf_exempt
def univariate_graphs_class(request, dataset_id):
    try:
        if request.method == 'POST':
            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])

            if dataset_data:
                df = pd.DataFrame(dataset_data)

                folder_path = os.path.join(settings.MEDIA_ROOT, 'univariate_graphs_class', dataset_id)
                os.makedirs(folder_path, exist_ok=True)

                categorical_columns = df.select_dtypes(include=['object']).columns

                for column in df.select_dtypes(include=['number']).columns:
                    plt.figure(figsize=(10, 6))
                    for category in categorical_columns:
                        sns.boxplot(x=df[category], y=df[column], data=df)
                    plt.title(f'Diagrama de caja para {column} por clase')
                    boxplot_path = os.path.join(folder_path, f'{column}_boxplot.png')
                    plt.savefig(boxplot_path)
                    plt.clf()

                for column in df.select_dtypes(include=['number']).columns:
                    plt.figure(figsize=(10, 6))
                    for category in categorical_columns:
                        sns.kdeplot(df[df[category].notna()][column], label=f'Clase {category}')
                    plt.title(f'Gráfico de densidad para {column} por clase')
                    density_plot_path = os.path.join(folder_path, f'{column}_density_plot.png')
                    plt.savefig(density_plot_path)
                    plt.clf()

                return JsonResponse({'mensaje': 'Gráficos generados y almacenados con éxito.', 'folder_path': folder_path})
            else:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    

