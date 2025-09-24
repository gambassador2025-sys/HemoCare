from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictRequestSerializer
from .predictor import HbPredictor
import threading

# Load predictor once (thread-safe lazy init)
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
    """
    POST /predict/
    JSON body: { "red": float, "ir": float, "gender": "male"/"female", "age": float }
    Returns: { "predicted_hb": float }
    """
    def post(self, request, *args, **kwargs):
        serializer = PredictRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        data = serializer.validated_data
        pred = get_predictor().predict(
            red=data['red'], ir=data['ir'],
            gender=data['gender'], age=data['age']
        )
        return Response({"predicted_hb": round(float(pred), 3)}, status=status.HTTP_200_OK)
