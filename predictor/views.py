# predictor/views.py (replace or update post method)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictRequestSerializer
from .predictor import HbPredictor
import threading

_predictor = None
_lock = threading.Lock()

def get_predictor():
    global _predictor
    if _predictor is None:
        with _lock:
            if _predictor is None:
                _predictor = HbPredictor(model_dir="./models")
    return _predictor

class PredictView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = PredictRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data

        debug_flag = False
        # allow debug via query param ?debug=1
        try:
            debug_flag = str(request.query_params.get("debug", "")).lower() in ("1", "true", "yes")
        except Exception:
            debug_flag = False

        try:
            result = get_predictor().predict(
                red=data['red'], ir=data['ir'],
                gender=data['gender'], age=data['age'],
                debug=debug_flag
            )
        except RuntimeError as e:
            return Response({"detail": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            return Response({"detail": "Prediction failed", "error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if debug_flag and isinstance(result, dict):
            # return full breakdown
            return Response(result, status=status.HTTP_200_OK)
        # else result is a float
        return Response({"predicted_hb": round(float(result), 3)}, status=status.HTTP_200_OK)
