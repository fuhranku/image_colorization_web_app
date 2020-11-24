from django.shortcuts import render

# Create your views here.
def index(request):
    # Renderiza la plantilla HTML index.html con los datos en la variable contexto
    return render(
        request,
        'index.html'
    )