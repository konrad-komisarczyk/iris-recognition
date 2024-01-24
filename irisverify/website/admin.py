from django.contrib import admin
from .models import WebsiteUser


class WebsiteUserAdmin(admin.ModelAdmin):
    list_display = ["user", "feature_vector", "unique_information"]


admin.site.register(WebsiteUser, WebsiteUserAdmin)
