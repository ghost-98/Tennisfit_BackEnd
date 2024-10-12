from django.contrib import admin
from django.urls import include, path
from django.http import HttpResponse

def index(request):
    return HttpResponse("Tennisfit 서버임.")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index),
    path('api/users/', include('users.urls')),
    path('api/similarity/', include('similarity.urls')),
]