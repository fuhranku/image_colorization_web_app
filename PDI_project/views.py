from django.shortcuts import render
from django.http import JsonResponse
from .computer_vision.imageProcesser import LogPrint, ProcessImage
import json

# Create your views here.
def index(request):
    return render(
        request,
        'index.html'
    )

def colorize(request):
    if request.is_ajax and request.method == "POST":
        # Process the image
        _request = request.FILES.get("uploadFile",None)
        return JsonResponse(
            {
                "Success":"Ajax Successful!",
                 "url":ProcessImage(_request,int(request.POST['option']))
            },
            status=200)

