import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view

# مسیرها را بررسی کن (ممکن است لازم باشد "../" را تغییر دهی)
MODEL_PATH = '../models/house_price_model.pkl'
COLUMNS_PATH = '../models/model_columns.pkl'

# بارگذاری مدل و لیست ستون‌ها
model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)


@api_view(['GET', 'POST'])
def predict(request):
    try:
        data = request.data
        df = pd.DataFrame([data])

        # تبدیل مقادیر بولین به عددی (۰ و ۱)
        for col in ['Parking', 'Warehouse', 'Elevator']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: 1 if str(x).lower() in [
                                        'true', '1', 'on', 'yes'] else 0)

        # one-hot encoding برای Address
        if 'Address' in df.columns:
            df = pd.get_dummies(df, columns=['Address'], drop_first=True)

        # اطمینان از وجود تمام ستون‌ها
        missing_cols = [col for col in model_columns if col not in df.columns]
        if missing_cols:
            df = pd.concat(
                [df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)

        # مرتب‌سازی ستون‌ها مطابق با مدل
        df = df[model_columns]

        # پیش‌بینی
        pred = model.predict(df)[0]

        return JsonResponse({'predicted_price': float(pred)})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
