from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import uuid


class WebsiteUser(models.Model):

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    feature_vector = models.TextField(max_length=500, blank=True, null=True)
    unique_information = models.UUIDField(default=uuid.uuid4, editable=False)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        WebsiteUser.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.websiteuser.save()


