from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('landing/', views.landing, name='landing'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('video_feed/<str:camera_id>/', views.video_feed, name='video_feed_cam'),
    path('api/counts/', views.get_counts, name='get_counts'),
    path('api/view-mode/', views.set_view_mode, name='set_view_mode'),
    path('api/zones/', views.manage_zones, name='manage_zones'),
    path('api/faces/known/', views.get_known_faces, name='get_known_faces'),
    path('api/faces/unknown/', views.get_unknown_faces, name='get_unknown_faces'),
    path('api/faces/masked/', views.get_masked_faces, name='get_masked_faces'),
    path('api/faces/stats/', views.get_face_stats, name='get_face_stats'),
    path('api/faces/inspect/', views.inspect_face, name='inspect_face'),
]
