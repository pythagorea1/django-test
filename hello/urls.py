from django.urls import path
from . import views
from .views import HelloView, LineChartJSONView, ChartData

urlpatterns = [
    path('next',views.next, name='next'),
    path('form',views.form, name='form'),
    path(r'', HelloView.as_view(), name='index'),
    path('api/chart/data', ChartData.as_view())
]
