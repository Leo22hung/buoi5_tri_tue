import pandas as pd 
import numpy as np 
np.random.seed(1)
#tom_tat
dc_listings = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',','')
stripped_dollar_signs = stripped_commas.str.replace('$','')
dc_listings['price'] = stripped_dollar_signs.astype(float)
print(dc_listings.info())
#loai_bo_dac_trung
drop_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode',
                'host_response_rate', 'host_acceptance_rate', 'host_listings_count']
dc_listings = dc_listings.drop(drop_columns, axis = 1)
print (dc_listings.isnull().sum()) 
#xu_li_gia_tri_bi_thieu
dc_listings = dc_listings.drop(['cleaning_fee', 'security_deposit'], axis=1)
dc_listings = dc_listings.dropna(subset=['bedrooms', 'bathrooms', 'beds'])
print(dc_listings.isnull().sum())
#chuan_hoa_cot
normalized_cod = (dc_listings['maximum_nights'] - dc_listings['maximum_nights'].mean()) / dc_listings['maximum_nights'].std() 
normalized_listings = (dc_listings - dc_listings.mean()) / (dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))
#khoang_cach_Euclidean_cho_truong_hop_da_bien
from scipy.spatial import distance
first_listing = [-0.596544, -0.439151]
second_listing = [-0.596544, -0.412923]
dist = distance.euclidean(first_listing, second_listing)
print(dist)

first_row = normalized_listings.loc[0, ['accommodates','bathrooms']]
fifth_row = normalized_listings.loc[4, ['accommodates','bathrooms']]
first_fifth_distance = distance.euclidean(first_row,fifth_row)
print(first_fifth_distance)
#huan_luyen_mo_hinh_va_du_doan
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors = 5, algorithm='brute')
train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]
train_features = train_df[['accommodates','bathrooms']]
train_target = train_df['price']
knn.fit(train_features,train_target)
test_features = test_df[['accommodates', 'bathrooms']]
predictions = knn.predict(test_features)
print(predictions)
#tinh_MSE_bang_Scikit-Learn
from sklearn.metrics import mean_squared_error
train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors = 5, algorithm='brute',metric='euclidean')
knn.fit(train_df[train_columns],train_df['price'])
predictions = knn.predict(test_df[train_columns])
two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = two_features_mse **0.5
print("Mean Squared Error (MSE):",two_features_mse)
print("Root Mean Squared Error (RMSE):",two_features_rmse)
#su_dung_tat_ca_cac_dac_trung
knn = KNeighborsRegressor(n_neighbors = 5, algorithm='brute')
features = train_df.columns.tolist()
features.remove('price')
knn.fit(train_df[features],train_df['price'])
all_features_predictions = knn.predict(test_df[features])
all_features_mse = mean_squared_error(test_df['price'], all_features_predictions)
all_features_rmse = all_features_mse**(1/2)
print(all_features_mse)
print(all_features_rmse)