from django.shortcuts import render
from django.http import JsonResponse
from .computer_vision.imageProcesser import LogPrint, ProcessImage

# Create your views here.
def index(request):
    # Renderiza la plantilla HTML index.html con los datos en la variable contexto
    return render(
        request,
        'index.html'
    )

def colorize(request):
    if request.is_ajax and request.method == "POST":
        # Process the image
        _request = request.FILES.get("img",None)
        
        return JsonResponse(
            {
                "Success":"Ajax Successful!",
                 "url":ProcessImage(_request)
            },
            status=200)

