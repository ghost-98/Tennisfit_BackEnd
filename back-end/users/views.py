from rest_framework import generics, status
from rest_framework.views import APIView
from .models import CustomUser, UserProfile
from .serializers import RegisterSerializer, LoginSerializer, LogoutSerializer, UsernameCheckSerializer, PhoneNumberCheckSerializer, FindUserIDSerializer, UserVerificationSerializer, PasswordResetSerializer, UserProfileUpdateSerializer
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.token_blacklist.models import BlacklistedToken # 내장 모델
from rest_framework_simplejwt.tokens import RefreshToken, TokenError
from rest_framework_simplejwt.views import TokenRefreshView
from django.contrib.auth import authenticate
from django.contrib.auth.hashers import make_password


# 회원가입 뷰
# generic클래스 상속받아서 crud(post)구현
class RegisterView(generics.CreateAPIView):
    permission_classes = [AllowAny]
    serializer_class = RegisterSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()  # 모델 인스턴스 db에 저장하고 user에 저장

            # 회원가입시 유저 프로필 인스턴스 아이디만 값 넣어서 생성
            UserProfile.objects.create(username=user)  # 외래키라 가능

            return Response({"message": "회원가입 성공!"}, status=status.HTTP_201_CREATED)
        else:
            # 에러 코드 및 메시지를 포함하여 반환
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 로그인 뷰
# 로그인 요청 처리
class LoginView(generics.GenericAPIView):
    permission_classes = [AllowAny]
    serializer_class = LoginSerializer

    # 로그인 요청(post)시에 호출됨
    # GenericAPIView에서 http기본메서드들 정의 하므로 오버라이드해서 씀
    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data) # 로그인 시리얼라이저 인스턴스화
        serializer.is_valid(raise_exception=True) # 데이터 검증

        # 사용자 인증
        user = authenticate(username=serializer.validated_data['username'],
                            password=serializer.validated_data['password'])

        # access, refresh 토큰 발급
        if user is not None:
            refresh_token = RefreshToken.for_user(user)  # 리프레시 토큰 발급

            user_profile = UserProfile.objects.get(username=user.username)

            return Response(
                {
                    "refresh": str(refresh_token),
                    "access": str(refresh_token.access_token),
                    "message": "로그인 성공",
                    "profile": {
                        "ntrp": user_profile.ntrp,
                        "career_period": user_profile.career_period,
                        "active_in_club": user_profile.active_in_club,
                        "one_line_introduction": user_profile.one_line_introduction,
                        "pro_amateur": user_profile.pro_amateur,
                        "profile_picture": request.build_absolute_uri(
                            user_profile.profile_picture.url) if user_profile.profile_picture else None,
                    }
                },
                status=status.HTTP_200_OK,
            )
        return Response({"error": "잘못된 사용자 이름 또는 비밀번호"}, status=status.HTTP_401_UNAUTHORIZED)


# 로그아웃 뷰
class LogoutView(generics.GenericAPIView):
    permission_classes = [IsAuthenticated] # allowany와 다르게 인증된 자만 권한. jwt토큰
    serializer_class = LogoutSerializer

    def post(self, request, *args, **kwargs):
        # 요청의 인증 정보 확인
        if request.auth:
            # 블랙리스트 모델에 추가
            BlacklistedToken.objects.create(token=request.auth)
            return Response(status=status.HTTP_205_RESET_CONTENT)
        return Response({"detail": "토큰이 없습니다."}, status=status.HTTP_400_BAD_REQUEST)


# django-allauth를 사용할 때, 소셜 로그인에 대한 뷰 설정은 allauth 내부에 구현되어 있음.

# 회원가입 중 아이디 중복 체크 뷰, 시리얼라이저가 아이디 유효성 검증
# 직관적으로 DB에서 체크해서 구현한 코드 아닌, 장고 내장기능 불러옴
class UsernameCheckView(generics.GenericAPIView):
    serializer_class = UsernameCheckSerializer
    permission_classes = [AllowAny] # 인증없이 권한을 누구에게나 줌. 아니면 뷰마다 혹은 세팅에서 전역으로 권한 설정해야함

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            return Response({"message": "사용 가능한 유저네임입니다."}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 전화번호 중복확인 뷰, 시리얼라이저가 전화번호 유효성 검증
class PhoneNumberCheckView(generics.GenericAPIView):
    serializer_class = PhoneNumberCheckSerializer
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        if serializer.is_valid():
            return Response({"message": "존재하는 전화번호입니다"}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 이름과 전화번호 이용해서 유저 아이디를 찾는 뷰
class FindUserIDView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = FindUserIDSerializer(data=request.data)
        if serializer.is_valid():
            return Response(serializer.validated_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 비밀번호 재설정 뷰 (아이디, 이름, 전화번호 확인후 재설정까지 한번에 기능하는)
'''class PasswordResetView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            new_password = serializer.validated_data['new_password']

            # 비밀번호 재설정
            user.password = make_password(new_password)
            user.save()

            return Response({"message": "비밀번호가 성공적으로 재설정되었습니다."}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)'''


# 비밀번호 재설정시 사용자 검증 뷰
class UserVerificationView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserVerificationSerializer(data=request.data)
        if serializer.is_valid():
            user_id = serializer.validated_data['user_id']
            # 검증된 사용자 ID를 응답으로 반환 (세션이나 프론트엔드에서 기억해야 함)
            return Response({"user_id": user_id}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 비밀번호 재설정시 비밀번호 재설정 뷰
class PasswordResetView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PasswordResetSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.validated_data['user']
            new_password = serializer.validated_data['new_password']

            # 비밀번호 재설정
            user.password = make_password(new_password)
            user.save()

            return Response({"message": "비밀번호가 성공적으로 재설정되었습니다."}, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 회원탈퇴 뷰 임시 : 탈퇴전에 비밀번호 받을지?
class DeleteUserView(APIView):
    permission_classes = [IsAuthenticated]  # 로그인된 사용자만 탈퇴 가능

    def delete(self, request):
        user = request.user  # 현재 로그인된 사용자
        user.delete()  # 유저 삭제
        return Response({"message": "회원탈퇴가 완료되었습니다."}, status=status.HTTP_204_NO_CONTENT)


# 프로필 수정 뷰. PUT 메소드
class UserProfileUpdateView(generics.UpdateAPIView):
    queryset = UserProfile.objects.all()  # queryset은 뷰가 작업할 모델 인스턴스 정의. db에있는 모든 인스턴스 가져옴 / 인스턴스 데이터들 필터링등 가능
    serializer_class = UserProfileUpdateSerializer  # 뷰와 연결할 serializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        user = self.request.user  # 현재 요청한 사용자
        user_profile = UserProfile.objects.get(username=user)
        return user_profile  # 프로필 인스턴스를 반환

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', True)  # 부분 업데이트 허용
        instance = self.get_object()  # 현재 사용자의 UserProfile 인스턴스
        serializer = self.get_serializer(instance, data=request.data, partial=partial)  # 받아온 요청이 업데이트 된 UserProfile 인스턴스

        if serializer.is_valid():
            self.perform_update(serializer)  # 인스턴스 업데이트
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# 유저의 액세스 토큰 만료시 재발급. 내장 tokenrefreshview 통해
class TokenRefreshView(TokenRefreshView):
    permission_classes = [AllowAny]

    def post(self, request, *args, **kwargs):
        refresh_token = request.data.get('refresh')

        if not refresh_token:
            return Response({"error": "리프레시 토큰이 필요합니다"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # 기존 리프레시 토큰 검증과 폐기
            old_refresh_token = RefreshToken(refresh_token)  # 토큰 문자열을 refreshtoken 객체로
            old_refresh_token.blacklist()  # 토큰 블랙리스트에 등록해 사용 불가능하게 함

            # 새 리프레시 토큰과 액세스 토큰 발급
            new_refresh_token = RefreshToken.for_user(old_refresh_token.user)
            access_token = new_refresh_token.access_token

            return Response(
                {
                    "refresh": str(new_refresh_token),  # 새 리프레시 토큰
                    "access": str(access_token),  # 새 액세스 토큰
                    "message": "새로운 토큰 발급 성공"
                },
                status=status.HTTP_200_OK
            )

        except TokenError:
            return Response({"error": "유효하지 않은 리프레시 토큰입니다"}, status=status.HTTP_401_UNAUTHORIZED)
