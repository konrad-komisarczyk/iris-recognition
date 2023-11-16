import traceback
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth.models import User
from iris_model.verify import verify


class IrisBackend(BaseBackend):
    def authenticate(self, request, username=None, iris_image=None, just_registered=False):
        try:
            user = User.objects.get(username=username)
            feature_vector = user.websiteuser.feature_vector
            if (feature_vector is not None and verify(iris_image, feature_vector)) or just_registered:
                return user
        except User.DoesNotExist:
            return None
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None