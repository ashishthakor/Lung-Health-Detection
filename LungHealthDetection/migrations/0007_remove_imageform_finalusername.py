# Generated by Django 3.2.8 on 2021-11-10 10:19

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('LungHealthDetection', '0006_auto_20211110_1515'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imageform',
            name='finalusername',
        ),
    ]