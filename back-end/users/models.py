from django.contrib.auth.models import AbstractUser
from django.db import models


# AbstractUser를 상속받아서 기본user 모델에서 제공하지 않는 속성 추가한 커스텀 모델
class CustomUser(AbstractUser):
    name = models.CharField(max_length=5)
    gender = models.CharField(max_length=5, null=True)
    birth_year = models.PositiveIntegerField(null=True, blank=True)
    birth_month = models.PositiveIntegerField(null=True, blank=True)
    birth_day = models.PositiveIntegerField(null=True, blank=True)
    phone_number = models.CharField(max_length=13, null=True, blank=True, unique=True)

    def __str__(self):  # for 디버깅
        return self.username


class UserProfile(models.Model):
    username = models.ForeignKey(
        CustomUser, to_field='username', on_delete=models.CASCADE, related_name='profiles')  # settings.AUTH_USER_MODEL로 참조 가능
    ntrp = models.FloatField(null=True, blank=True)
    career_period = models.FloatField(null=True, blank=True)
    active_in_club = models.CharField(max_length=100, null=True, blank=True)
    one_line_introduction = models.CharField(max_length=255, null=True, blank=True)
    pro_amateur = models.BooleanField(default=False)  # Professional(True) or amateur(False)
    profile_picture = models.ImageField(upload_to='profile_pics/', null=True, blank=True)

    def __str__(self):
        return self.username
