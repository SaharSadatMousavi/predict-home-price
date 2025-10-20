from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib

model = joblib.load('predictor/house_price_model.pkl')

@api_view(['POST'])
def predict(request):
    data = request.data
    area = float(data.get('area'))
    location = data.get('location')
    rooms = int(data.get('rooms'))

    # اینجا ورودی‌ها رو میدیم به مدل:
    prediction = model.predict([[area, rooms]])[0]

    return Response({'predicted_price': prediction})
