from django.contrib.auth.forms import UsernameField
from django.db import models
# from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.conf import settings
from django.http import request

User = settings.AUTH_USER_MODEL

# Create your models here.


class ImageForm(models.Model):
    userid = models.ForeignKey(User, default = 1, null = True, on_delete = models.CASCADE)
    # finalusername = models.CharField(max_length=265,null=True,blank=True)
    photo = models.ImageField(upload_to="myimage")
    date = models.DateTimeField(auto_now_add=True)
    data = models.TextField(null=True,blank = True)

# exapmle image model for xyz
class Image(models.Model):
    photo = models.ImageField(upload_to="myimage")
    date = models.DateTimeField(auto_now_add=True)
    data = models.TextField(null=True,blank =True)