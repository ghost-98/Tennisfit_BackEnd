from django.urls import path
from .views import VideoUploadView, VideoProcessView, SwingAnalyzeView

urlpatterns = [
    path('upload_video/', VideoUploadView.as_view(), name='upload_video'),
    path('process_video/', VideoProcessView.as_view(), name='extract_video'),
    path('analyze_swing/', SwingAnalyzeView.as_view(), name='analyze_swing'),
]