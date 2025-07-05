from django.db import models
import uuid

class PredictionLog(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    # Storing the input features as a JSON object
    input_data = models.JSONField()

    investment_probability = models.FloatField()
    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id} at {self.created_at}"