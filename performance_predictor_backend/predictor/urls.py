# backend/myapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('process-player-data/', views.process_player_data, name='process_player_data'),
    # Add other URLs as needed
]
