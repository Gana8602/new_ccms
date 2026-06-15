from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('api/counts/', views.get_counts, name='get_counts'),
    path('api/view-mode/', views.set_view_mode, name='set_view_mode'),
    path('api/faces/known/', views.get_known_faces, name='get_known_faces'),
    path('api/faces/unknown/', views.get_unknown_faces, name='get_unknown_faces'),
    path('api/faces/masked/', views.get_masked_faces, name='get_masked_faces'),
    path('api/faces/stats/', views.get_face_stats, name='get_face_stats'),
]
