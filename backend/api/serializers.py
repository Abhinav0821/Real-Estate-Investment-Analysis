from rest_framework import serializers

class PredictionInputSerializer(serializers.Serializer):
    Area = serializers.FloatField()
    Floor = serializers.IntegerField()
    Num_Bedrooms = serializers.IntegerField(min_value=0)
    Num_Bathrooms = serializers.IntegerField(min_value=0)
    Property_Age = serializers.IntegerField(min_value=0)
    Proximity = serializers.FloatField()
    image = serializers.ImageField()