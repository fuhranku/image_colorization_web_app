from django.db import models

# Create your models here.
class ImageModel(models.Model):
    title  = models.CharField(max_length=100, default="output.jpg")  # this field does not use in your project
    image  = models.FileField(upload_to='img/')
    width  = models.IntegerField(null=True)
    height = models.IntegerField(null=True)