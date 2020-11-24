from django.shortcuts import render

from .computer_vision.test import log_print

# Create your views here.
def index(request):
    log_print()
    # Renderiza la plantilla HTML index.html con los datos en la variable contexto
    return render(
        request,
        'index.html'
    )