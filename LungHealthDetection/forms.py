from django.forms import ModelForm, fields
from django.contrib.auth.forms import UserCreationForm, UserChangeForm
from django.contrib.auth.models import User,UserManager
from django import forms
from django.db import models
from .models import Image,ImageForm




# this will not use in this project for now
# class OrderForm(ModelForm):
# 	class Meta:
# 		model = Order
# 		fields = '__all__'

class CreateUserForm(UserCreationForm):
	class Meta:
		model = User
		fields = ['username', 'email', 'password1', 'password2']

class UpdateUserForm(UserChangeForm):
	class Meta:
		model = User
		fields = ['first_name', 'last_name']

class FinalImage(forms.ModelForm):
	class Meta:
		model = ImageForm
		# fields = '__all__'
		fields = ['photo']
		labels = {'userid':"UserName"}


# Example Image Form for xyz
class ImageForm(forms.ModelForm):
	class Meta:
		model = Image
		fields = '__all__'
		labels = {'photo':''}