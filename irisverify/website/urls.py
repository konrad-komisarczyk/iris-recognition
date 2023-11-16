from django.urls import path

from .views import index, login_view, logout_view, register_view, secrets

urlpatterns = [
    path("", index, name='index'),
    path("secrets/", secrets, name='secrets_view'),
    path("login/", login_view, name='login_view'),
    path("logout/", logout_view, name='logout_view'),
    path("register/", register_view, name='register_view'),
]