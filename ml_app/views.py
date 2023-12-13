import json
from uuid import uuid4
from bson import ObjectId
from django.conf import settings
import pandas as pd
from django.http import JsonResponse
from pymongo import MongoClient
from django.views.decorators.csrf import csrf_exempt
from ml_app.database import get_database_connection
from ml_app.utils import load_best_model, load_dataset, store_imputed_dataset,load_datasetImputation, convert_to_serializable
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import os
import pickle



from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
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

            describe_result = df.describe().to_dict()
            describe_result = json.loads(json.dumps(describe_result, default=convert_to_serializable))

            return JsonResponse(describe_result, json_dumps_params={'indent': 2})
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
            dataset = load_dataset(dataset_id)
            dataset_data = dataset.get('data', [])

            if dataset_data:
                df = pd.DataFrame(dataset_data)

                folder_path = os.path.join(settings.MEDIA_ROOT, 'univariate_graphs', dataset_id)

                os.makedirs(folder_path, exist_ok=True)

                images_paths = []
                for column in df.columns:
                    plt.figure(figsize=(8, 6))

                    if df[column].dtype == 'object':
                        # Histograma para variables categóricas
                        plt.figure(figsize=(8, 6))
                        sns.countplot(x=df[column])
                        plt.title(f'Count Plot for {column}')

                    elif np.issubdtype(df[column].dtype, np.number):
                        # Histograma para variables numéricas
                        plt.figure(figsize=(8, 6))
                        sns.histplot(df[column], kde=False)
                        plt.title(f'Histogram for {column}')

                        # Diagrama de caja para variables numéricas
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x=df[column])
                        plt.title(f'Boxplot for {column}')

                        # Gráfico de distribución de probabilidad
                        plt.figure(figsize=(8, 6))
                        sns.distplot(df[column])
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

            potential_id_columns = ['No', 'Id', 'codigo', 'PassengerId']  
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
                encoded_data = encoder.fit_transform(df[categorical_columns].fillna('Missing'))  
                numerical_df = pd.concat([numerical_df, pd.DataFrame(encoded_data)], axis=1)

            # Convertir los nombres de las columnas a cadenas
            numerical_df.columns = numerical_df.columns.astype(str)

            pca_model = PCA()
            transformed_data = pca_model.fit_transform(numerical_df)
            components_weights = pca_model.components_

            new_dataset_id = str(uuid4())
            
            for i, record in enumerate(transformed_data):
                new_dataset_document = {
                    '_id': f"{new_dataset_id}_{i}",
                    'original_dataset_id': dataset_id,
                    'data': {f'component_{j}': record[j] for j in range(len(record))}
                }
                new_collection = db["DatasetsPCA"]
                new_collection.insert_one(new_dataset_document)

            return JsonResponse({
                'mensaje': 'PCA aplicado con éxito',
                'new_dataset_id': new_dataset_id,
                'data': pd.DataFrame(transformed_data, columns=[f'component_{i}' for i in range(transformed_data.shape[1])]).to_dict(orient='records')
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

            if not dataset_data:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)

            df = pd.DataFrame(dataset_data)

            folder_path = os.path.join(settings.MEDIA_ROOT, 'univariate_graphs_class', dataset_id)
            os.makedirs(folder_path, exist_ok=True)

            categorical_columns = df.select_dtypes(include=['object']).columns
            numeric_columns = df.select_dtypes(include=['number']).columns

            for column in numeric_columns:
                # Imprime los datos para verificar si son válidos
                print(f'Datos para {column}: {df[column].dropna().tolist()}')

                # Gráfico de densidad
                plt.figure(figsize=(10, 6))
                sns.kdeplot(df[column].dropna())
                plt.title(f'Gráfico de densidad para {column}')
                density_plot_path = os.path.join(folder_path, f'{column}_density_plot.png')
                plt.savefig(density_plot_path)
                plt.clf()

                # Diagrama de caja
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=df[column].dropna())
                plt.title(f'Diagrama de caja para {column}')
                box_plot_path = os.path.join(folder_path, f'{column}_box_plot.png')
                plt.savefig(box_plot_path)
                plt.clf()

            return JsonResponse({'mensaje': 'Gráficos generados y almacenados con éxito.', 'folder_path': folder_path})
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'}, status=405)
    except Exception as e:
        return JsonResponse({'error': str(e)})
    
@csrf_exempt
def train_models(request, dataset_id):
    try:
        if request.method == 'POST':
            data = json.loads(request.body)
            dataset = load_datasetImputation(dataset_id)
            dataset_data = dataset.get('data', [])

            if not dataset_data:
                return JsonResponse({'error': 'El dataset no contiene datos'}, status=400)

            df = pd.DataFrame(dataset_data)
            df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
            algorithms = data.get('algorithms', [])
            option_train = int(data.get('option_train', 0))
            normalization = int(data.get('normalization', 0))
            target_column = data.get('target_column', None)

            if target_column is None or target_column not in df.columns:
                columns_available = df.columns.tolist()
                return JsonResponse({'error': 'Columna objetivo no válida o no especificada', 'columns_available': columns_available}, status=400)

            folder_path = os.path.join(settings.MEDIA_ROOT, 'train', dataset_id)
            os.makedirs(folder_path, exist_ok=True)
            trained_models = []

            for algorithm in algorithms:
                model = None
                if algorithm == 1:
                    model = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
                elif algorithm == 2:
                    model = KNeighborsClassifier()
                elif algorithm == 3:
                    model = SVC(class_weight='balanced', random_state=42)
                elif algorithm == 4:
                    model = GaussianNB()
                elif algorithm == 5:
                    model = DecisionTreeClassifier(random_state=42)
                elif algorithm == 6:
                    model = MLPClassifier(max_iter=1000, random_state=42)

                if normalization == 1:
                    scaler = MinMaxScaler()
                elif normalization == 2:
                    scaler = StandardScaler()
                else:
                    return JsonResponse({'error': 'Tipo de normalización no válido. Utilice 1 o 2.'}, status=400)

                if option_train == 1:
                    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), df[target_column], test_size=0.3, random_state=42)
                    scaler.fit(X_train)  
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                    confusion = confusion_matrix(y_test, predictions).tolist()
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(y_test, predictions)
                    recall = recall_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions)
                elif option_train == 2:
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    min_samples = min(df[target_column].value_counts())
                    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                    predictions = cross_val_predict(model, scaler.fit_transform(X), y, cv=kf)
                    confusion = confusion_matrix(y, predictions).tolist()
                    accuracy = accuracy_score(y, predictions)
                    precision = precision_score(y, predictions)
                    recall = recall_score(y, predictions)
                    f1 = f1_score(y, predictions)
                else:
                    return JsonResponse({'error': 'Opción de entrenamiento no válida. Utilice 1 o 2.'}, status=400)

                algorithm_mapping = {
                    1: 'Regresión Logística',
                    2: 'KNN',
                    3: 'Máquinas de soporte vectorial',
                    4: 'Naive Bayes',
                    5: 'Árboles de decisión',
                    6: 'Redes neuronales multicapa',
                }

                training_id = str(uuid4())

                model_filename = os.path.join(folder_path, f"{training_id}_model.pkl")
                with open(model_filename, 'wb') as model_file:
                    pickle.dump(model, model_file)

                trained_model_info = {
                    '_id': training_id,
                    'original_dataset_id': dataset_id,
                    'algorithm': algorithm_mapping.get(algorithm),
                    'target_column': target_column,
                    'confusion_matrix': confusion,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
                filtered_trained_model_info = {
                    '_id': training_id,
                    'original_dataset_id': dataset_id,
                    'algorithm': algorithm_mapping.get(algorithm),
                    'target_column': target_column,
                }

                client, db = get_database_connection()
                collection_name = "TrainedModels"
                collection = db[collection_name]
                collection.insert_one(trained_model_info)
                trained_models.append(filtered_trained_model_info)

            return JsonResponse({'mensaje': 'Entrenamiento realizado con éxito', 'training_id': training_id, 'trained_models': trained_models})
        else:
            return JsonResponse({'error': 'Solicitud no válida. Debe ser una solicitud POST'}, status=405)

    except Exception as e:
        return JsonResponse({'error': str(e)})


@csrf_exempt
def results(request, train_id):
    try:
        client, db = get_database_connection()
        collection_name = "TrainedModels"
        collection = db[collection_name]
        trained_model_info = collection.find_one({'_id': train_id})

        if not trained_model_info:
            return JsonResponse({'error': 'Modelo entrenado no encontrado'}, status=404)

        results = {
            'algorithm': trained_model_info.get('algorithm'),
            'confusion_matrix': trained_model_info.get('confusion_matrix'),
            'accuracy': trained_model_info.get('accuracy'),
            'precision': trained_model_info.get('precision'),
            'recall': trained_model_info.get('recall'),
            'f1_score': trained_model_info.get('f1_score'),
        }

        return JsonResponse(results, json_dumps_params={'indent': 2})

    except Exception as e:
        return JsonResponse({'error': str(e)})
    