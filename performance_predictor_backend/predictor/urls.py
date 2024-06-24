from django.urls import path
from . import views

urlpatterns = [
    path('process-player-data/', views.process_player_data, name='process_player_data'),
    path('predict/', views.predict_score, name='predict_score'),
    path('get-player-id/', views.get_player_id, name='get_player_id')
    # Add other URLs as needed
]
