from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import PredictionInputSerializer
from .services.predictor import prediction_service
from .models import PredictionLog

class PredictionView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        serializer = PredictionInputSerializer(data=request.data)
        if serializer.is_valid():
            validated_data = serializer.validated_data
            image_file = validated_data.pop('image') # Separate image from other data

            try:
                # Get prediction from our service
                probability = prediction_service.predict(validated_data, image_file)

                # Log the prediction
                log_entry = {k: float(v) for k, v in validated_data.items()}
                PredictionLog.objects.create(
                    input_data=log_entry,
                    investment_probability=probability
                )

                return Response({'investment_probability': probability}, status=status.HTTP_200_OK)

            except Exception as e:
                return Response({"error": f"Model prediction failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)