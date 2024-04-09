from django.urls import path
from . import views

urlpatterns = [
    path('ml', views.getData),
    path('vType', views.getVType),
    path('vMake', views.getVMake),
    path('vUsage', views.getVUsage)
]
