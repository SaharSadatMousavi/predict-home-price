import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view

# مسیرها را با ساختار پروژه‌ات هماهنگ کن
MODEL_PATH = '../../models/house_price_model.pkl'
COLUMNS_PATH = '../../models/model_columns.pkl'

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

@api_view(['POST'])
def predict(request):
    try:
        data = request.data
        df = pd.DataFrame([data])

        # اگر ستون‌های بولی وجود دارند و به int نیاز دارن:
        for col in ['Parking', 'Warehouse', 'Elevator']:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # one-hot برای Address (همانند نوتبوک)
        if 'Address' in df.columns:
            df = pd.get_dummies(df, columns=['Address'], drop_first=True)

        # مطابقت دادن ستون‌ها با ستون‌های آموزش:
        for col in model_columns:
            if col not in df.columns:
                df[col] = 0

        # ترتیب درست:
        df = df[model_columns]

        pred = model.predict(df)[0]

        return JsonResponse({'predicted_price': float(pred)})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
