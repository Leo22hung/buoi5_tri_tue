import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')
test_df = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')


two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms'] 
hyper_params = list(range(1, 21))

two_mse_values = []
three_mse_values = []
two_hyp_mse = {}
three_hyp_mse = {}

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    mse = mean_squared_error(test_df['price'], predictions)
    two_mse_values.append(mse)

two_lowest_mse = min(two_mse_values)
two_lowest_k = two_mse_values.index(two_lowest_mse) + 1 
two_hyp_mse[two_lowest_k] = two_lowest_mse

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])  
    mse = mean_squared_error(test_df['price'], predictions)
    three_mse_values.append(mse)

three_lowest_mse = min(three_mse_values)
three_lowest_k = three_mse_values.index(three_lowest_mse) + 1 
three_hyp_mse[three_lowest_k] = three_lowest_mse

print("Two Features - Optimal k and MSE:", two_hyp_mse)
print("Three Features - Optimal k and MSE:", three_hyp_mse)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hyper_params, two_mse_values, marker='o')
plt.title('MSE vs. Number of Neighbors (Two Features)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(hyper_params, three_mse_values, marker='o', color='orange')
plt.title('MSE vs. Number of Neighbors (Three Features)')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid()

plt.tight_layout()
plt.show()