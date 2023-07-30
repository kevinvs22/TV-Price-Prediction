import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
tv_data = pd.read_csv('Dataset.csv')

# Define the price categories
price_categories = {'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4}

# Map the prices to the categories
tv_data['Price_Category'] = pd.cut(tv_data['Price'], bins=[499, 624, 749, 874, 1000], labels=['C1', 'C2', 'C3', 'C4'])
tv_data['Price_Category'] = tv_data['Price_Category'].map(price_categories)

# Encode the 'Type' column
le = LabelEncoder()
tv_data['Type'] = le.fit_transform(tv_data['Type'])

# Define the feature and target variables
X = tv_data.drop(['Price', 'Price_Category'], axis=1)
y = tv_data['Price_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical variables
scaler = MinMaxScaler()
X_train[['Display Size', 'Refresh Rate', 'Resolution']] = scaler.fit_transform(X_train[['Display Size', 'Refresh Rate',
                                                                                        'Resolution']])
X_test[['Display Size', 'Refresh Rate', 'Resolution']] = scaler.transform(X_test[['Display Size', 'Refresh Rate',
                                                                                  'Resolution']])

# Create the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan')

# Train the classifier using the training data
knn.fit(X_train, y_train)

# Predict the categories for the test data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Evaluate the precision of the model
precision = precision_score(y_test, y_pred, average='weighted')
print('Precision:', precision)

# Evaluate the recall of the model
recall = recall_score(y_test, y_pred, average='weighted')
print('Recall:', recall)

# Evaluate the F1 score of the model
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)

# Perform k-fold cross-validation
scores = cross_val_score(knn, X, y, cv=15)

# Print the mean score and standard deviation
print('Cross-validation mean score:', scores.mean())
print('Cross-validation standard deviation:', scores.std())
