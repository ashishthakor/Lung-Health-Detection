from django.contrib import admin
from django.db.models import fields
from .models import ImageForm
from django.contrib.auth.models import User
from .models import Image
# Register your models here.

# @admin.register(ImageForm)
# class ImageFormAdmin(admin.ModelAdmin):
#     list_display = ['id', 'userid', 'photo','date','data']
    # class Meta:
    #     model = ImageForm
    #     fields = '__all__'
# allmodels = [ImageForm,ImageFormAdmin]
# admin.site.register(ImageForm, ImageFormAdmin)

@admin.register(ImageForm)
class ImageFormAdmin(admin.ModelAdmin):
    list_display = ['id','userid','photo','date','data']

# exapmle image admin form/model
@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'photo', 'date','data']

