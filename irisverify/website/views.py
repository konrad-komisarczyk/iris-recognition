from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
import traceback
import logging
from .forms import WebsiteUserLoginForm, WebsiteUserCreationForm
from .models import WebsiteUser
from iris_model.verify import extract_feature_vector

logger = logging.getLogger("irisverify")


# Create your views here.


def index(request):
    return render(request, 'home.html')


@login_required(login_url="/login")
def secrets(request):
    user = User.objects.get(pk=request.user.id)
    return render(request, "secrets.html", {"secret": str(user.websiteuser.unique_information)})


def login_view(request):
    login_error = False
    user_exists = True
    if request.method == 'POST':
        form = WebsiteUserLoginForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data["username"]
            iris_image = form.cleaned_data["iris_image"]
            if username_exists(username):
                try:
                    user = authenticate(username=username, iris_image=iris_image)
                except Exception as e:
                    logger.error(f'Error when authenticating user:\n {traceback.format_exc()}')
                    user = None

                if user is not None:
                    login(request, user)
                    return redirect('index')
                else:
                    login_error = True
            else:
                user_exists = False
        else:
            login_error = True
    else:
        form = WebsiteUserLoginForm()
        login_error = False
    return render(request, "login.html", {"form": form, "login_error": login_error, "user_exists": user_exists})


def logout_view(request):
    logout(request)
    return redirect('login_view')


def username_exists(username):
    return WebsiteUser.objects.filter(user__in=User.objects.filter(username=username).all()).exists()


def register_view(request):
    if request.method == "POST":
        form = WebsiteUserCreationForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data["username"]
            iris_image = form.cleaned_data["iris_image"]
            if not username_exists(username):
                user_tmp = User(username=username)
                user_tmp.save()
                websiteuser = WebsiteUser.objects.get(user=user_tmp)
                try:
                    websiteuser.feature_vector = extract_feature_vector(iris_image)
                    websiteuser.save()
                except Exception as e:
                    logger.error(f'Error when extracting feature vector from input image:\n {traceback.format_exc()}')
                    user_tmp.delete()
                    return render(request, "register.html", {"form": form, "extraction_error": True})
                user = authenticate(username=username, iris_image=iris_image, just_registered=True)
                login(request, user)
                return redirect("index")
            else:
                return render(request, "register.html", {"form": form, "username_taken": True})
    form = WebsiteUserCreationForm()
    return render(request, "register.html", {"form": form})
