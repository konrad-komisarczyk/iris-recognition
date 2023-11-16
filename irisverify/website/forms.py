from django import forms


class WebsiteUserLoginForm(forms.Form):
    username = forms.CharField()
    iris_image = forms.ImageField()

    def __init__(self, *args, **kwargs):
        super(WebsiteUserLoginForm, self).__init__(*args, **kwargs)
        self.fields['username'].label = "Nazwa użytkownika:"
        self.fields['iris_image'].label = "Zdjęcie tęczówki:"


class WebsiteUserCreationForm(forms.Form):
    username = forms.CharField()
    iris_image = forms.ImageField()

    def __init__(self, *args, **kwargs):
        super(WebsiteUserCreationForm, self).__init__(*args, **kwargs)
        self.fields['username'].label = "Nazwa użytkownika:"
        self.fields['iris_image'].label = "Zdjęcie tęczówki:"
