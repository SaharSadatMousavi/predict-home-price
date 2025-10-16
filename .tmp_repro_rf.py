import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

p='d:/vc code/house_price_project/data/cleaned_housePrice_enhanced.csv'
df=pd.read_csv(p)
print('Loaded df shape:', df.shape)

# mimic notebook preprocessing
y = df['Price']
X = df.drop(columns=['Price', 'Price_Million', 'Price_per_sqm', 'Price_per_sqm_Million'])

# bool cols
bool_cols = ['Parking', 'Warehouse', 'Elevator']
for c in bool_cols:
    if c in X.columns:
        print(c, 'dtype before:', X[c].dtype, 'unique sample:', X[c].dropna().unique()[:5])
        X[c] = X[c].astype(int)

# dtypes
print('\nDtypes value counts:')
print(X.dtypes.value_counts())

# any object columns
obj_cols = X.select_dtypes(include=['object']).columns.tolist()
print('\nObject columns sample (first 20):', obj_cols[:20])

print('\nNull counts (top 20):')
print(X.isnull().sum().loc[lambda s: s>0].sort_values(ascending=False).head(20))

# get dummies for Address if present
if 'Address' in X.columns:
    X = pd.get_dummies(X, columns=['Address'], drop_first=True)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('\nAfter split shapes:', X_train.shape, X_test.shape)

# attempt fit
try:
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    print('Training succeeded')
except Exception as e:
    print('Exception during fit:')
    import traceback; traceback.print_exc()
    print('\nSample dtypes in X_train:')
    print(X_train.dtypes.head(30))
    print('\nAny object columns in X_train? ', X_train.select_dtypes(include=['object']).columns.tolist())
    print('\nAny NaNs in X_train? ', X_train.isnull().any().any())
    if X_train.isnull().any().any():
        print(X_train.isnull().sum().loc[lambda s: s>0].head(20))
