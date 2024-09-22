import pandas as pd
import matplotlib.pyplot as plt
train_df = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')
test_df = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
features = ['accommodates', 'bedrooms', 'bathrooms','number_of_reviews']
hyper_params = [x for x in range(1,21)]
mse_values = []
features = train_df.columns.tolist()
for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors = hp,algorithm='brute')
    knn.fit(train_df[features],train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
plt.scatter(hyper_params, mse_values)
plt.title('MSE vs. Number of Neighbors')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid()
plt.show()
print(mse_values)

