# Generated by Django 3.1.3 on 2020-11-25 23:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('PDI_project', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='title',
            field=models.CharField(default='output.jpg', max_length=100),
        ),
    ]
