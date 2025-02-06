from django.urls import path
from .views import predictor, formInfo

urlpatterns = [
    path("", predictor, name="home"),
    path("formInfo/", formInfo, name="predict"),
]