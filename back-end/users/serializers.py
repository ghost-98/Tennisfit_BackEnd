# serializer는 DRF 설계원칙에 따라 주로 데이터를 검증(유효성), 직렬화/역직렬화 맡음
# ModelSerializer과 Serializer는 다름

from rest_framework import serializers
from django.contrib.auth.models import User
from django.contrib.auth import get_user_model
from django.contrib.auth.hashers import check_password
from .models import UserProfile

User = get_user_model()

# 회원가입 요청 받으면 json파일을
# 회원가입시 사용자 데이터 검증 및 새로운 사용자 생성
class RegisterSerializer(serializers.ModelSerializer): # 상속받는 저 클래스는 받은 json파일을 db user에 맞게 데이터변환
    class Meta:
        model = User
        fields = ['username', 'password',
                  'gender', 'name', 'phone_number',
                  'birth_year', 'birth_month', 'birth_day']
        extra_kwargs = {
            'password': {'write_only': True} # 클라이언트는 응답에서 비밀번호 볼 수 x
        }

    # 전화번호 중복 체크
    def validate_phone_number(self, value):
        if User.objects.filter(phone_number=value).exists():
            raise serializers.ValidationError("해당 전화번호는 이미 사용 중입니다.")
        return value

    '''    # 비밀번호 자릿수 '유효성 검사'"
        def validate(self, attrs):
            if 'password' in attrs and len(attrs['password']) < 8:
                raise serializers.ValidationError({"password": "비밀번호는 8자리 이상이어야 합니다."})
            return attrs'''

    def create(self, validated_data):
        user = User.objects.create_user( # 이렇게 db 접근해서 인스턴스 생성하는것
            username=validated_data['username'], # 속성
            password=validated_data['password'],
            name=validated_data['name'], # 여기까지 3개는 user모델에서 기본제공
            gender=validated_data['gender'],
            phone_number=validated_data['phone_number'],
            birth_year=validated_data['birth_year'],
            birth_month=validated_data['birth_month'],
            birth_day=validated_data['birth_day'],
        )
        return user

# 로그인 시리얼라이저
class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True)

class LogoutSerializer(serializers.Serializer):
    # 로그아웃 시 필요한 데이터가 있다면 여기에 정의
    pass

# 회원가입 중 아이디 중복 시리얼라이저
class UsernameCheckSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)

    # 아이디 유효성 검증
    def validate_username(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError({"username": "유저네임이 이미 존재합니다."})
        return value

# 전화번호 중복확인 시리얼라이저
class PhoneNumberCheckSerializer(serializers.Serializer):
    phone_number = serializers.CharField(max_length=15)

    # 전화번호 유효성 검중
    def validate_phone_number(self, value):
        if User.objects.filter(phone_number=value).exists():
            raise serializers.ValidationError({"phone_number": "전화번호가 이미 존재합니다."})
        return value

# 아이디 찾기 시리얼라이저 (이름, 전화번호)
# 시리얼라이저로 받은 데이터 유효성 검사 및 DB에 접근하여 쿼리로 값 찾음
class FindUserIDSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    phone_number = serializers.CharField(max_length=15)

    def validate(self, data):
        name = data.get('name')
        phone_number = data.get('phone_number')
        # DB에서 이름과 전화번호 모두 일치하는 사용자 찾는 쿼리
        try:
            user = User.objects.get(name=name, phone_number=phone_number)
            return {'username': user.username}
        except User.DoesNotExist:
            raise serializers.ValidationError("해당 사용자 정보를 찾을 수 없습니다.")

# 비밀번호 재설정 시리얼라이저 (아이디, 이름, 전화번호 확인후 재설정까지 한번에 기능하는)
'''class PasswordResetSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    username = serializers.CharField(max_length=150)
    phone_number = serializers.CharField(max_length=15)
    new_password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True, min_length=8)

    def validate(self, data):
        name = data.get('name')
        username = data.get('username')
        phone_number = data.get('phone_number')
        new_password = data.get('new_password')
        confirm_password = data.get('confirm_password')

        # 이름, 아이디, 전화번호가 일치하는 사용자 찾기
        try:
            user = User.objects.get(name=name, username=username, phone_number=phone_number)
        except User.DoesNotExist:
            raise serializers.ValidationError("해당 사용자 정보를 찾을 수 없습니다.")

        # 새 비밀번호와 비밀번호 확인이 일치하는지 확인
        if new_password != confirm_password:
            raise serializers.ValidationError("새 비밀번호와 비밀번호 확인이 일치하지 않습니다.")

        # 새 비밀번호가 기존 비밀번호와 동일한지 확인
        if check_password(new_password, user.password):
            raise serializers.ValidationError("새 비밀번호는 기존 비밀번호와 같을 수 없습니다.")

        # 유효성 검사를 통과하면 사용자와 새 비밀번호를 반환
        return {
            'user': user,
            'new_password': new_password
        }
'''

# 비밀번호 재설정시 사용자 검증 api 시리얼라이저
class UserVerificationSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    name = serializers.CharField(max_length=100)
    phone_number = serializers.CharField(max_length=15)

    def validate(self, data):
        username = data.get('username')
        name = data.get('name')
        phone_number = data.get('phone_number')

        # 이름, 아이디, 전화번호가 일치하는 사용자 찾기
        try:
            user = User.objects.get(username=username, name=name, phone_number=phone_number)
        except User.DoesNotExist:
            raise serializers.ValidationError("해당 사용자 정보를 찾을 수 없습니다.")

        # 사용자 검증에 성공하면 해당 사용자의 ID를 반환
        return {'user_id': user.id}

# 비밀번호 재설정시 비밀번호 재설정 api 시리얼라이저
class PasswordResetSerializer(serializers.Serializer):
    username = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True, min_length=8)
    # confirm_password = serializers.CharField(write_only=True, min_length=8)

    def validate(self, data):
        username = data.get('username')
        new_password = data.get('new_password')
        # confirm_password = data.get('confirm_password')

        # 사용자 검증
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise serializers.ValidationError("사용자를 찾을 수 없습니다.")

        '''# 새 비밀번호와 비밀번호 확인이 일치하는지 확인
        if new_password != confirm_password:
            raise serializers.ValidationError("새 비밀번호와 비밀번호 확인이 일치하지 않습니다.")'''

        # 새 비밀번호가 기존 비밀번호와 동일한지 확인
        if check_password(new_password, user.password):
            raise serializers.ValidationError("새 비밀번호는 기존 비밀번호와 같을 수 없습니다.")

        return {'user': user, 'new_password': new_password}


# 모델 시리얼라이저는 orm 모델과 직접연결되어 유효성 검증, 직렬화 자동 처리
class UserProfileUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['ntrp', 'career_period', 'active_in_club', 'one_line_introduction', 'pro_amateur', 'profile_picture']
        # 각 필드에 추가 설정 적용 (DRF에서 serializer 정의 시)
        extra_kwargs = {
            'ntrp': {'required': False},
            'career_period': {'required': False},
            'active_in_club': {'required': False},
            'one_line_introduction': {'required': False},
            'pro_amateur': {'required': False},
            'profile_picture': {'required': False}
        }
