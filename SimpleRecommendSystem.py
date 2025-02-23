import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', 500)

items = ['Book', 'Pen', 'Pencil', 'BackPack', 'Paper', 'Eraser']

data = {
    'Customer': ['C1', 'C2', 'C3', 'C4', 'C5'],
    'Orders': [
        ['Book', 'Pen', 'Pencil', 'Paper', 'Eraser'],
        ['BackPack', 'Pen'],
        ['Paper', 'Eraser'],
        ['Book', 'BackPack', 'Paper'],
        ['Pencil', 'Eraser', 'Pen', 'BackPack']
    ]
}

df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(np.array(items).reshape(-1, 1))

encoded_list = []

for item in df['Orders']:
    encoded = encoder.transform(np.array(item).reshape(-1, 1))
    customer_encoded = np.sum(encoded, axis=0)
    encoded_list.append(customer_encoded)

df['EncodedOrders'] = encoded_list

new_customer_order = ['Book', 'BackPack', 'Paper']
new_customer_order_encoded = encoder.transform(np.array(new_customer_order).reshape(-1, 1))
new_customer_order_encoded = np.sum(new_customer_order_encoded, axis=0)

similarity = np.round(cosine_similarity([new_customer_order_encoded], np.vstack(df['EncodedOrders'])), 2)
df['Similarities'] = similarity.reshape(-1, 1)


def recommended_items():
    df.drop(df[df['Similarities'] == 1].index, inplace=True)
    sorted_df = df.sort_values(by='Similarities', ascending=False)
    print(sorted_df)
    most_similar = sorted_df['Orders'].iloc[0]

    recommends = [x for x in most_similar if x not in new_customer_order]

    return recommends


print('Recommended Items:', recommended_items())