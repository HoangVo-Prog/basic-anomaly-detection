import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

iris = load_iris()
data = iris.data  
features = iris.feature_names

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

model = IsolationForest(contamination=0.05, random_state=42) 
model.fit(data_scaled)

# Predict anomalies (-1 = anomaly, 1 = normal)
predictions = model.predict(data_scaled)


x1 = data[:, 0]  
x2 = data[:, 1]  

plt.figure(figsize=(10, 6))
plt.scatter(x1, x2, c=predictions, cmap='coolwarm', label='Anomalies')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Anomaly Detection on Iris Dataset')
plt.legend(['Normal', 'Anomalous'])
plt.show()

print("Detected Anomalies (Indexes of anomalies in the dataset):")
anomalies = np.where(predictions == -1)[0]
print(anomalies)