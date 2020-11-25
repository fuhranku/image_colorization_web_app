from django.shortcuts import render
from django.http import JsonResponse
from .computer_vision.test import log_print

# Create your views here.
def index(request):
    log_print(1)
    # Renderiza la plantilla HTML index.html con los datos en la variable contexto
    return render(
        request,
        'index.html'
    )

def colorize(request):
    log_print(2)
    if request.is_ajax and request.method == "POST":
        return JsonResponse({"Success":"Ajax Successful!"},status=200)