from rest_framework import serializers

class PredictRequestSerializer(serializers.Serializer):
    red = serializers.FloatField()
    ir = serializers.FloatField()
    gender = serializers.ChoiceField()
    age = serializers.FloatField(min_value=0, max_value=130)

    def validate_gender(self, value):
        v = str(value).strip().lower()
        if v in ['m','male']:
            return 'male'
        if v in ['f','female']:
            return 'female'
        raise serializers.ValidationError("Gender must be male/female")

class PredictResponseSerializer(serializers.Serializer):
    predicted_hb = serializers.FloatField()
