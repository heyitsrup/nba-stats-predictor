# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import subprocess
import json

@csrf_exempt
@require_POST
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
            output = result.stdout.strip().splitlines()  # Example: process output if needed
            return JsonResponse({'success': True, 'output': output})
        else:
            error_message = result.stderr.strip().splitlines()  # Example: handle error messages
            return JsonResponse({'error': 'Failed to process player data', 'details': error_message}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
