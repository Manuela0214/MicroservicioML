from django.contrib import admin
from django.urls import path
from ml_app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    # Otras rutas
    path('', views.home, name='home'),
    path('load/', views.load, name='load'),
    path('basic statistics/<str:dataset_id>/', views.basic_statistics, name='basic_statistics'),
]
