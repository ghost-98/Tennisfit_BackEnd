from rest_framework import serializers
from .models import Video


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['video_file']


# 작명
class VideoProcessSerializer(serializers.ModelSerializer):
    video_id = serializers.CharField(required=True)
    selected_object_id = serializers.IntegerField(required=True)
    frame_at_return = serializers.IntegerField(required=True)

    class Meta:
        model = Video
        fields = ['video_id', 'selected_object_id', 'frame_at_return']


class SwingAnalyzeSerializer(serializers.ModelSerializer):
    video_id = serializers.CharField(required=True)
    class Meta:
        model = Video
        fields = ['video_id']