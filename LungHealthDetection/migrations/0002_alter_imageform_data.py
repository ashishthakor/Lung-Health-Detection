# Generated by Django 3.2.8 on 2021-11-08 11:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('LungHealthDetection', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageform',
            name='data',
            field=models.TextField(null=True),
        ),
    ]