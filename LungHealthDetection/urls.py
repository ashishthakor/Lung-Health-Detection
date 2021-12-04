from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
# from LungHealthDetection.views import ResetPasswordView
# from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name="home"),
    path('home/', views.home, name="home"),
    path('dashboard/', views.dashboard, name="dashboard"),
    path('profile/', views.profile, name="profile"),
    path('records/', views.records, name="records"),
    path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),  
	path('logout/', views.logoutUser, name="logout"),
    path('predict/', views.predict, name = 'predict'),
    path('record_del/(?P<pk>\d+)/$', views.record_del, name = 'record_del'),
    path('user_del/(?P<pk>\d+)/$', views.user_del, name = 'user_del'),
    # path('predictxyz/', views.predictxyz, name = 'predictxyz'),
    # path('password_reset/', ResetPasswordView.as_view(), name='password_reset'),
    # path('password-reset-confirm/<uidb64>/<token>/',
    #     auth_views.PasswordResetConfirmView.as_view(template_name='LungHealthDetection/password_reset_confirm.html'),
    #     name='password_reset_confirm'),
    # path('password-reset-complete/',
    #     auth_views.PasswordResetCompleteView.as_view(template_name='LungHealthDetection/password_reset_complete.html'),
    #     name='password_reset_complete'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)