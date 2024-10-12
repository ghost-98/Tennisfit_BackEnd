from django.urls import path
from .views import (RegisterView, LoginView, LogoutView, UsernameCheckView, FindUserIDView, PhoneNumberCheckView,
                    UserVerificationView, PasswordResetView, DeleteUserView, UserProfileUpdateView, TokenRefreshView)
from django.http import HttpResponse

def index(request):
    return HttpResponse("api/users/ 루트임")

urlpatterns = [
    path('', index, name='index'),
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('check_username/', UsernameCheckView.as_view(), name='check-username'),
    path('check_phone_number/',  PhoneNumberCheckView.as_view(), name='check_phone_number'),
    path('find_username/',FindUserIDView.as_view(), name='find_username'),
    path('user_verification/', UserVerificationView.as_view(), name='user_verification'),
    path('reset_password/', PasswordResetView.as_view(), name='reset_password'),
    path('delete_user/', DeleteUserView.as_view(), name='delete_user'),
    path('update_user_profile/', UserProfileUpdateView.as_view(), name='update_user_profile'),
    path('token_refresh/', TokenRefreshView.as_view(), name='token_refresh'),
]
