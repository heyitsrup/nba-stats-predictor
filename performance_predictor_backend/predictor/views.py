from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import subprocess
import json
import logging

logger = logging.getLogger(__name__)

@csrf_exempt  # For demonstration purposes, disable CSRF protection (not recommended for production)
@require_POST  # Ensure view only responds to POST requests
def process_player_data(request):
    try:
        data = json.loads(request.body)
        player_name = data.get('player_name')
        if not player_name:
            return JsonResponse({'error': 'Player name is required'}, status=400)

        # Run Python script using subprocess
        result = subprocess.run(
            ['python', 'get_stats.py', '--player_name', player_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            return JsonResponse({'success': True})
        else:
            logger.error('Subprocess failed with return code %d: %s', result.returncode, result.stderr)
            return JsonResponse({'error': 'Failed to process player data'}, status=500)

    except json.JSONDecodeError:
        logger.error('Invalid JSON received: %s', request.body)
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        logger.error('Error processing player data: %s', str(e))
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Method not allowed'}, status=405)
