# Generated by Django 3.2.8 on 2021-11-09 13:26

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('LungHealthDetection', '0004_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imageform',
            name='userid',
            field=models.ForeignKey(default=1, null=True, on_delete=django.db.models.deletion.SET_NULL, to=settings.AUTH_USER_MODEL),
        ),
    ]
