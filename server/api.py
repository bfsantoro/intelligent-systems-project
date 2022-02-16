from flask import Flask
from flask import request
import os
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import json

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import SGDClassifier

app = Flask(__name__)
   
@app.route('/v1/categorize', methods=['POST'])
def categorize():
    all_categories = list()    
    
    request_products = request.json
    
    for product in request_products["products"]:
        try:
            product_category = classify_product(product)
        except:
            return {}, 400
        
        all_categories.append(product_category)
    
    dict_categories = {"categories":all_categories}
    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(dict_categories, f, ensure_ascii=False, indent=4)

    return {"categories":all_categories}

def classify_product(new_product):
    x_values = np.array([new_product[key] for key in ["price", "weight", "minimum_quantity", "view_counts"]])
    x_values_scaled = scaler.transform(x_values.reshape(1,-1))

    x_text = np.array(new_product["concatenated_tags"])
    x_text = x_text.reshape(1,-1)
    x_text_encoded = oh_encoder.transform(x_text)
    print(f"Resultado encoded: {x_text_encoded}")

    x_test = np.concatenate((x_values_scaled, x_text_encoded), axis=1)
    y_test = model.predict(x_test)
    y_test_decoded = encoder.inverse_transform(y_test)[0]
    print(y_test)
    print(y_test_decoded)
    return y_test_decoded

DATASET_PATH = os.environ['DATASET_PATH']
MODEL_PATH = os.environ['MODEL_PATH']

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv(DATASET_PATH)

df = df.dropna(subset=['price', 'weight', 'minimum_quantity', 'category', 'view_counts'])
outlier_filter = (abs(stats.zscore(df.price)<3))
df = df[outlier_filter]

indexes = [df.columns.get_loc(col) for col in ['price', 'weight','minimum_quantity', 'view_counts']]
values = df.values[:,indexes]

scaler = StandardScaler().fit(values)
values_scaled = scaler.transform(values)

texts = np.array(df['concatenated_tags'].tolist())
texts = texts.reshape(-1, 1)

oh_encoder = OneHotEncoder(sparse=False, handle_unknown = 'ignore')
oh_encoder.fit_transform(texts)

encoder = LabelEncoder()
y = df.values[:,-1]
y_encoded = encoder.fit_transform(y.ravel())

app.run(debug=False)