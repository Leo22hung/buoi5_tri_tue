import pandas as pd
import numpy as np

dc_listings = pd.read_csv('C:/Users/Admin/Downloads/dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',','')

stripped_dollars = stripped_commas.str.replace('$','')
dc_listings['price'] = stripped_dollars.astype('float')
shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)
split_one = dc_listings.iloc[0:1862].copy() 
split_two = dc_listings.iloc[1862:].copy() 


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one
model = KNeighborsRegressor()
model.fit(train_one[["accommodates"]], train_one["price"]) 

test_one["predicted_price"] = model.predict(test_one[["accommodates"]])
iteration_one_rmse = mean_squared_error(test_one["price"],
test_one["predicted_price"])**(1/2)
model.fit(train_two[["accommodates"]], train_two["price"]) 
test_two["predicted_price"] = model.predict(test_two[["accommodates"]])
iteration_two_rmse = mean_squared_error(test_two["price"], test_two["predicted_price"])**(1/2)
avg_rmse = np.mean([iteration_two_rmse, iteration_one_rmse])
print(iteration_one_rmse, iteration_two_rmse, avg_rmse)


dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
dc_listings.loc[dc_listings.index [1490:2234], "fold"] = 3
dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
dc_listings.loc[dc_listings.index [2978:3723], "fold"] = 5
print(dc_listings['fold'].value_counts())
print("\n Num of missing values: ", dc_listings['fold'].isnull().sum())

model = KNeighborsRegressor()
train_iteration_one = dc_listings[dc_listings["fold"] != 1] 
test_iteration_one = dc_listings[dc_listings["fold"] == 1].copy() 
model.fit(train_iteration_one[["accommodates"]], train_iteration_one["price"])
labels = model.predict(test_iteration_one [["accommodates"]])
test_iteration_one["predicted_price"] = labels

iteration_one_mse = mean_squared_error(test_iteration_one["price"],
test_iteration_one["predicted_price"])
iteration_one_rmse = iteration_one_mse ** (1/2)

print(iteration_one_mse)
print(test_iteration_one)
print(iteration_one_rmse)


fold_ids = [1, 2, 3, 4, 5]
def train_and_validate(df, folds):
    fold_rmses = [] 
    for fold in folds:
        model = KNeighborsRegressor() 
        train = df[df["fold"] != fold] 
        test = df[df["fold"] == fold].copy() 
        model.fit(train[["accommodates"]], train["price"]) 
        
        labels = model.predict(test[["accommodates"]])
        test["predicted_price"] = labels 
        
        mse = mean_squared_error(test["price"], test["predicted_price"])
        rmse = mse**(1/2)
        fold_rmses.append(rmse) 
    return(fold_rmses)

rmses = train_and_validate(dc_listings, fold_ids)
print(rmses) 

avg_rmse = np.mean(rmses)
print(avg_rmse) 
