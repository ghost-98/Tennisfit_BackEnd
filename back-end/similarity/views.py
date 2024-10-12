from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import VideoSerializer, VideoProcessSerializer, SwingAnalyzeSerializer
from .extracting_video import process_video
from .models import Video
from .analyze_swing import analyze_swing
import numpy as np
import torch


# 폼데이터 형식으로 요청
class VideoUploadView(APIView):
    permission_classes = [IsAuthenticated]
    video_list = []  # 동영상 ID를 저장하는 리스트

    def post(self, request, *args, **kwargs):
        video_serializer = VideoSerializer(data=request.data)

        if video_serializer.is_valid():
            video_instance = video_serializer.save()  # 비디오 인스턴스 DB에 저장
            # 고유 ID를 리스트에 추가
            self.video_list.append(str(video_instance.video_id))

            video_path = video_instance.video_file.path

            initial_output = process_video(video_path, [0, None])

            return Response({
                'video_id': video_instance.video_id,
                'frame': initial_output.get('frame'),
                'encoded_images': initial_output.get('encoded_images')
            }, status=status.HTTP_201_CREATED)

        return Response(video_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# json으로 요청받음
class VideoProcessView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # 요청받은 데이터 직렬화 및 유효성 검증
        video_serializer = VideoProcessSerializer(data=request.data)
        if video_serializer.is_valid():
            # 모델에서 video_id로 비디오 객체를 가져옴
            video_instance = Video.objects.get(video_id=video_serializer.validated_data['video_id'])
            video_path = video_instance.video_file.path  # 비디오 파일 경로 얻기

            start_frame = video_serializer.validated_data['frame_at_return']
            selected_id = video_serializer.validated_data['selected_object_id']

            # 이전에 처리된 데이터셋을 로드 (존재하지 않으면 빈 리스트로)
            prev_dataset_path = f"./media/processed_videos/{video_instance.video_id}_np.npy"
            prev_dataset = None
            try:
                # Numpy 파일에서 이전 데이터 불러오기
                prev_dataset = np.load(prev_dataset_path, allow_pickle=True).tolist()
            except FileNotFoundError:  # 파일이 없을 경우
                prev_dataset = []

            # 비디오 처리
            result = process_video(video_path, [start_frame, selected_id], prev_dataset=prev_dataset)

            # 이전 np파일을 연결해서 작업한 np파일을 저장
            np.save(prev_dataset_path, result['dataset'])

            # np 파일 경로를 모델에 저장
            video_instance.np_file_path = prev_dataset_path  # np 파일 경로 업데이트
            video_instance.save()

            # 중간에 멈추면
            if result.get('encoded_images'):
                return Response({
                    'video_id': video_instance.video_id,  # 비디오 ID
                    'frame': result.get('frame'),  # 현재 프레임
                    'encoded_images': result.get('encoded_images')  # base64 인코딩된 이미지 문자열 데이터
                }, status=status.HTTP_202_ACCEPTED)  # http 코드 맞는지?

            # 영상이 끝나면
            return Response({'message': 'Video processing completed.'}, status=status.HTTP_201_CREATED)  # http 코드 맞는지?

        # 직렬화 오류 시
        return Response(video_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SwingAnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        # 요청받은 데이터 직렬화 및 유효성 검증
        video_serializer = SwingAnalyzeSerializer(data=request.data)
        if video_serializer.is_valid():
            video_instance = Video.objects.get(video_id=video_serializer.validated_data['video_id'])
            np_file_path = video_instance.np_file_path

            final_result = analyze_swing(np_file_path)

            # DB에 저장하려면 numpy 배열을 리스트로 바꿔서 저장가능
            final_result_list = final_result

            # final_result DB에 저장
            video_instance.final_result = final_result_list
            video_instance.save()

            return Response({'result': final_result}, status=status.HTTP_200_OK)  # 200 코드가 맞나
        return Response(video_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
