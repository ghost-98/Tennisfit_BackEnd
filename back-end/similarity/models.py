from django.db import models
import uuid


# 비디오 모델 / uuid, FileField, 영상 저장 경로 지정
class Video(models.Model):
    video_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    video_file = models.FileField(upload_to='videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    np_file_path = models.CharField(max_length=255, blank=True, null=True)
    final_result = models.JSONField(blank=True, null=True)  # JSON필드? models 속성 보기

    def __str__(self):
        return str(self.video_id)
