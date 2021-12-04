from django.shortcuts import render, redirect 
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm


from django.contrib.auth import authenticate, login, logout

from django.contrib import messages

from django.contrib.auth.decorators import login_required

from django.urls import reverse_lazy
# from django.contrib.auth.views import PasswordResetView
from django.contrib.messages.views import SuccessMessageMixin

# Create your views here.
from .models import *
from .forms import CreateUserForm, FinalImage, UpdateUserForm,User
# from .filters import OrderFilter

from django.core.files.storage import FileSystemStorage

from .forms import ImageForm,FinalImage
from .models import ImageForm
from django.views.generic import ListView

from LungHealthDetection import final_miniProject as fp

# CreateUserForm => class name

# class recordView(ListView):
# 	model = ImageForm
# 	template_name = 'LungHealthDetection/records.html'
def user_del(request,pk):
	# print(pk)
	user = User.objects.filter(id = pk)
	# image = ImageForm.objects.get(userid = user.username)
	# image.delete()
	user.delete()
	return redirect('login')
def record_del(request,pk):
	# print(pk)
	record = ImageForm.objects.get(id = pk)
	record.delete()
	# redirect('records')
	return redirect('records')

def records(request):
	# objects = models.Manager()
	records = ImageForm.objects.all().order_by('-date')
	# print(records[0])
	return render(request, 'LungHealthDetection/records.html',{'records' : records })

def predict(request):
	output_list0 = []
	output_list1 = []
	output_list2 = []
	if request.method == "POST":
		form = FinalImage(request.POST, request.FILES)
		print(request.FILES)
		if form.is_valid():
			instance = form.save(commit=False)
			instance.userid = request.user
			instance.save()
			path = instance.photo.url
			output_list = fp.image_prediction_and_visualization('C:/Users/Ashish Thakor/Desktop/mini_project/mini_project/' + path)
			instance.data = output_list
			instance.save()
			output_list0  =output_list[0]
			output_list1  =output_list[1]
			output_list2  =output_list[2]
			# print(output_list[0])
			# print(output_list[1])
			# print(output_list[2])
			# print(output_list)
	form = FinalImage()
	# img = ImageForm.objects.all()
	return render(request, 'LungHealthDetection/predict.html', {'form':form,'output_list0':output_list0,'output_list1':output_list1,'output_list2':output_list2})

# def predictxyz(request):
# 	if request.method == "POST":
# 		form = ImageForm(request.POST, request.FILES)
# 		if form.is_valid():
# 			form.save()
# 	form = ImageForm()
# 	img = Image.objects.all()
# 	return render(request, 'LungHealthDetection/xyz.html', {'img':img, 'form':form})

def registerPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		form = CreateUserForm()
		if request.method == 'POST':
			form = CreateUserForm(request.POST)
			if form.is_valid():
				form.save()
				user = form.cleaned_data.get('username')
				messages.success(request, 'Account was created for ' + user)
				form = CreateUserForm()
				return redirect('login')
			

		context = {'form':form}
		return render(request, 'LungHealthDetection/register.html', context)

def loginPage(request):
	if request.user.is_authenticated:
		return redirect('home')
	else:
		if request.method == 'POST':
			username = request.POST.get('username')
			password =request.POST.get('password')

			user = authenticate(request, username=username, password=password)

			if user is not None:
				login(request, user)
				return redirect('home')
			else:
				messages.info(request, 'Username OR password is incorrect')

		context = {}
		return render(request, 'LungHealthDetection/login.html', context)

def logoutUser(request):
	logout(request)
	return redirect('login')

def home(request):
	context = {}
	return render(request, 'LungHealthDetection/index.html', context)

# @login_required
def dashboard(request):
	context = {}
	return render(request, 'LungHealthDetection/dashboard.html',context)  

# def profile(request):
# 	context = {}
# 	return render(request, 'LungHealthDetection/profile.html',context)  

def profile(request):
	if request.method == 'POST':
			form = UpdateUserForm(request.POST, instance = request.user)
			if form.is_valid():
				form.save()
				messages.success(request, 'Congratulations... Account Updated')
				return redirect('profile')
	else:
		form = UpdateUserForm(instance = request.user)
	context = {'form': form}
	return render(request,"LungHealthDetection/profile.html",context)







# def predict(request):
	# print(request.POST.dict())
	# print(request.POST.dict().get('filepath'))
	# print(request.FILES['filepath'])
	# fileobj = request.FILES.getlist('filepath')
	# fileobj = request.POST.dict().get('filepath')
	# print(fileobj)
	# fs = FileSystemStorage()
	# fs.save(fileobj)
	# context = {'img':fileobj}
	# if request.method == "POST":
	# 	form = ImageForm(request.POST, request.FILES)
	# 	if form.is_valid():
   	# 		form.save()
 	# form = ImageForm()
 	# img = ImageForm.objects.all()
	# context = { 'img':img, 'form':form }
	# context={}
	# return render(request, 'LungHealthDetection/predict.html',context)