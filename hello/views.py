from django.shortcuts import render
from django.http import HttpResponse
from .models import Friend

# Create your views here.
def index(request):
    data = Friend.objects.all()
    params = {
        'title':'Hello',
        'message':'all friends.',
        'data':data,
    }
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
