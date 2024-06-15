from django.urls import path
from . import views

urlpatterns = [
    path('process-player-data/', views.process_player_data, name='process_player_data'),
    path('predict/', views.predict_score, name='predict_score'),
    # Add other URLs as needed
]
