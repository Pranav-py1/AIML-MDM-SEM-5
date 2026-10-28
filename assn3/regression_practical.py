import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load the dataset
data = pd.read_csv("house_price.csv")
print("First 5 records:")
print(data.head())
print("\nDataset information:")
data.info()

# 2. Select input features and target
X = data[["Area", "Bedrooms", "Age"]]
y = data["Price"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train standard Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# 5. Calculate Linear Regression metrics
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, linear_pred)

print("\n==============================")
print("LINEAR REGRESSION PERFORMANCE")
print("==============================")
print(f"MAE      : {linear_mae:.4f}")
print(f"MSE      : {linear_mse:.4f}")
print(f"RMSE     : {linear_rmse:.4f}")
print(f"R2 Score : {linear_r2:.4f}")

# 6. Prepare arrays for Gradient Descent
Xtr = X_train.to_numpy(dtype=float)
Xte = X_test.to_numpy(dtype=float)
ytr = y_train.to_numpy(dtype=float).reshape(-1, 1)
yte = y_test.to_numpy(dtype=float).reshape(-1, 1)

# 7. Standardize features using training data
x_mean = Xtr.mean(axis=0)
x_std = Xtr.std(axis=0)
x_std[x_std == 0] = 1
Xtr = (Xtr - x_mean) / x_std
Xte = (Xte - x_mean) / x_std

# 8. Add intercept column
Xtr = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
Xte = np.c_[np.ones((Xte.shape[0], 1)), Xte]

# 9. Initialize Gradient Descent
theta = np.zeros((Xtr.shape[1], 1))
rate = 0.01
iterations = 5000
m = len(ytr)
cost_history = []

# 10. Train using Gradient Descent
for i in range(iterations):
    pred = Xtr @ theta
    error = pred - ytr
    cost = np.sum(error ** 2) / (2 * m)
    cost_history.append(cost)
    grad = (Xtr.T @ error) / m
    theta = theta - rate * grad

# 11. Generate predictions
# Gradient Descent predictions are in the same units as Price
gd_pred = Xte @ theta

# 12. Evaluate Gradient Descent
# Convert arrays to 1-D for clean metric output
gd_mae = mean_absolute_error(yte.ravel(), gd_pred.ravel())
gd_mse = mean_squared_error(yte.ravel(), gd_pred.ravel())
gd_rmse = np.sqrt(gd_mse)
gd_r2 = r2_score(yte.ravel(), gd_pred.ravel())

print("\n==============================")
print("GRADIENT DESCENT PERFORMANCE")
print("==============================")
print(f"Final Cost: {cost_history[-1]:.4f}")
print(f"MAE      : {gd_mae:.4f}")
print(f"MSE      : {gd_mse:.4f}")
print(f"RMSE     : {gd_rmse:.4f}")
print(f"R2 Score : {gd_r2:.4f}")

# 13. Show predictions for test samples
pred_table = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Linear Prediction": linear_pred,
    "GD Prediction": gd_pred.ravel()
})
print("\nPREDICTIONS ON TEST DATA")
print(pred_table.round(2).to_string(index=False))

# 14. Plot actual and predicted prices
plt.figure(figsize=(7, 5))
plt.scatter(y_test, linear_pred)
plt.xlabel("Actual Price (lakh)")
plt.ylabel("Predicted Price (lakh)")
plt.title("Linear Regression: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 15. Plot cost convergence
plt.figure(figsize=(7, 5))
plt.plot(range(1, iterations + 1), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Gradient Descent Cost vs Iterations")
plt.grid(True)
plt.tight_layout()
plt.show()

# 16. Plot Gradient Descent predictions
plt.figure(figsize=(7, 5))
plt.scatter(y_test, gd_pred.ravel())
plt.xlabel("Actual Price (lakh)")
plt.ylabel("Predicted Price (lakh)")
plt.title("Gradient Descent: Actual vs Predicted")
plt.grid(True)
plt.tight_layout()
plt.show()

# 17. Compare model metrics
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Gradient Descent"],
    "MAE": [linear_mae, gd_mae],
    "MSE": [linear_mse, gd_mse],
    "RMSE": [linear_rmse, gd_rmse],
    "R2 Score": [linear_r2, gd_r2]
})
print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

plt.figure(figsize=(7, 5))
plt.bar(comparison["Model"], comparison["RMSE"])
plt.ylabel("RMSE")
plt.title("Model Comparison - RMSE")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.bar(comparison["Model"], comparison["R2 Score"])
plt.ylabel("R2 Score")
plt.title("Model Comparison - R2 Score")
plt.grid(axis="y")
plt.tight_layout()
plt.show()
