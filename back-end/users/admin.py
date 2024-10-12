from django.contrib import admin
from .models import CustomUser

# 관리자 페이지에서 커스텀한 유저 모델 관리
admin.site.register(CustomUser)