from sklearn import datasets
from sklearn.cluster import KMeans

# Loading Dataset
iris_df = datasets.load_iris()

# Declaring Model
model = KMeans(n_clusters=3)

# Fitting model
model.fit(iris_df.data)

# Predicting a single input
predicted_label = model.predict([[7.2, 3.5, 0.8, 1.6]])

# Prediction on the entire data
all_predictions = model.predict(iris_df.data)

#Printing Predictions
print(predicted_label)
print(all_predictions)
