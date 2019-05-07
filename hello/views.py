from django.shortcuts import render
from django.http import HttpResponse
from .models import Friend
import dynamic_portfolio as d
from django.views.generic import TemplateView
from chartjs.views.lines import BaseLineChartView
from .forms import Options
from rest_framework.views import APIView
from rest_framework.response import Response
from django.contrib.auth import get_user_model
import os, json, pickle
# Create your views here.

class LineChartJSONView(BaseLineChartView):
    def get_labels(self):
        """Return 7 labels for the x-axis."""
        return ["January", "February", "March", "April", "May", "June", "July"]

    def get_providers(self):
        """Return names of datasets."""
        return ["Central", "Eastside", "Westside"]

    def get_data(self):
        """Return 3 datasets to plot."""

        return [[75, 44, 92, 11, 44, 95, 35],
                [41, 92, 18, 3, 73, 87, 92],
                [87, 21, 94, 3, 90, 13, 65]]

class HelloView(TemplateView):

    def __init__(self):
        self.params = {
            'title':'Optimal Portfolio',
            'form':Options()
        }

    def get(self, request):
        return render(request, 'hello/index.html', self.params)

    def post(self, request):
        print(request.POST)
        try:
            self.params = d.main(request.POST['base'],request.POST['timespan'],request.POST['number_of_portfolio'])
        except Exception:
            self.params = dict()
        self.params['form'] = Options(request.POST)
        print("parameters")
        print(self.params)
        return render(request, 'hello/index.html', self.params)
def index(request):
    params = d.main()
    return render(request, 'hello/index.html', params)

def next(request):
    params = {
        'title':'Hello/Next',
        'msg':'This is another page.',
        'goto':'index',
    }
    return render(request, 'hello/index.html', params)

def form(request):
    msg = request.POST['msg']
    params = {
        'title':'Hello/Form',
        'msg':'Hello : '+msg,
    }
    return render(request, 'hello/index.html', params)

class ChartData(APIView):
    authentication_classes = []
    permission_classes = []
    def get(self, request, format=None):
        if os.path.exists("hello/static/result.json"):
            with open("hello/static/result.json") as f:
                data = json.loads(f.read())
        else:
            data = dict()
        return Response(data)
