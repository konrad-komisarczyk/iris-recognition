from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from .forms import WebsiteUserLoginForm, WebsiteUserCreationForm
from .models import WebsiteUser
from iris_model.verify import extract_feature_vector

# Create your views here.


def index(request):
    return render(request, 'home.html')


@login_required(login_url="/login")
def secrets(request):
    user = User.objects.get(pk=request.user.id)
    return render(request, "secrets.html", {"secret": str(user.websiteuser.unique_information)})


def login_view(request):
    user = None
    if request.method == 'POST':
        form = WebsiteUserLoginForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data["username"]
            iris_image = form.cleaned_data["iris_image"]

            user = authenticate(username=username, iris_image=iris_image)

            if user is not None:
                login(request, user)
                return redirect('index')
        login_error = True
    else:
        form = WebsiteUserLoginForm()
        login_error = False
    return render(request, "login.html", {"form": form, "login_error": login_error})


def logout_view(request):
    logout(request)
    return redirect('login_view')


def register_view(request):
    if request.method == "POST":
        form = WebsiteUserCreationForm(request.POST, request.FILES)
        if form.is_valid():
            username = form.cleaned_data["username"]
            iris_image = form.cleaned_data["iris_image"]
            user_tmp = User(username=username)
            user_tmp.save()
            websiteuser = WebsiteUser.objects.get(user=user_tmp)
            websiteuser.feature_vector = extract_feature_vector(iris_image)
            websiteuser.save()
            user = authenticate(username=username, iris_image=iris_image, just_registered=True)
            login(request, user)
            return redirect("index")
    form = WebsiteUserCreationForm()
    return render(request=request, template_name="register.html", context={"form": form})
