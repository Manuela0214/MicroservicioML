from django.contrib import admin
from django.urls import path
from ml_app import views


urlpatterns = [
    path('admin/', admin.site.urls),
    # Otras rutas
    path('', views.home, name='home'),
    path('load/', views.load, name='load'),
    path('basic statistics/<str:dataset_id>/', views.basic_statistics, name='basic_statistics'),
    path('columns-describe/<str:dataset_id>/', views.columns_describe, name='columns-describe'),
    path('imputation/<str:dataset_id>/type/<str:number_type>/', views.imputation, name='imputation'),
    path('general-univariate-graphs/<str:dataset_id>', views.general_univariate_graphs, name='general_univariate_graphs'),
    path('univariate-graphs-class/<str:dataset_id>/', views.univariate_graphs_class, name='univariate_graphs_class'),
    path('bivariate-graphs-class/<str:dataset_id>/', views.bivariate_graphs_class, name='bivariate_graphs_class'),
    path('multivariate-graphs-class/<str:dataset_id>/', views.multivariate_graphs_class, name='multivariate_graphs_class'),
    path('pca/<str:dataset_id>/', views.pca, name='pca'),    
    path('train/<str:dataset_id>/', views.train_models, name='train_models'),
    path('results/<str:train_id>/', views.results, name='get_results'),
]
