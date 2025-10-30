import joblib
import pandas as pd
from django.http import JsonResponse
from rest_framework.decorators import api_view
from pathlib import Path
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

MODEL_PATH = Path(settings.BASE_DIR) / "models" / "house_price_model.pkl"
COLUMNS_PATH = Path(settings.BASE_DIR) / "models" / "model_columns.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    if not isinstance(model_columns, (list, tuple)):
        model_columns = list(model_columns)
except Exception as e:
    logger.exception("Error loading model or model_columns: %s", e)
    model = None
    model_columns = []


@api_view(['GET', 'POST'])
def predict(request):
    try:
        incoming = request.data if request.method == 'POST' else request.query_params
       

        df = pd.DataFrame([incoming])

        # پاک‌سازی نام ستون‌ها
        df.columns = df.columns.str.strip()

        # تبدیل ایمن مقادیر بولین (بدون apply برای کارایی بهتر)
        # تبدیل امن و بدون تکه‌تکه کردن DataFrame
        bool_cols = ['Parking', 'Warehouse', 'Elevator']
        bool_data = {}
        for col in bool_cols:
            val = incoming.get(col, False)
            if isinstance(val, bool):
                bool_data[col] = [int(val)]
            elif isinstance(val, str):
                bool_data[col] = [1 if val.lower() in ['true','1','on','yes','y','checked'] else 0]
            else:
                bool_data[col] = [0]

        df_bool = pd.DataFrame(bool_data)
        df = pd.concat([df.drop(columns=bool_cols, errors='ignore'), df_bool], axis=1)


        # one-hot برای Address
        if 'Address' in df.columns:
            df = pd.get_dummies(df, columns=['Address'], drop_first=False)

        # فقط ستون‌هایی که مدل آموزش دیده را نگه دار
        # اضافه کردن ستون‌های مورد نیاز مدل که در df نیستن
        missing_cols = [col for col in model_columns if col not in df.columns]
        if missing_cols:
            df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, df_missing], axis=1)

        # بازآرایی نهایی ستون‌ها مطابق مدل
        df = df.reindex(columns=model_columns, fill_value=0)

        
        # پیش‌بینی
        if model is None:
            return JsonResponse({'error': 'Model not loaded on server'}, status=500)

        pred = model.predict(df)[0]
        return JsonResponse({'predicted_price': float(pred)})

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return JsonResponse({'error': str(e)}, status=400)