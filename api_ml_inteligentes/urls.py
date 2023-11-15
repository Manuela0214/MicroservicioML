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
]
