import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # <-- For saving the model

# 1. Load your dataset
df = pd.read_csv('data3.csv')  # replace with your actual dataset path

# 2. Separate features and target
X = df.drop('V', axis=1)
y = df['V']

# 3. Encode the target column using LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. (Optional) Encode categorical features if needed
# X = pd.get_dummies(X)  # if X has categorical columns

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 6. Train Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# 7. Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 8. Save the model and label encoder
joblib.dump(clf, 'decision_tree_model.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model and Label Encoder saved as 'decision_tree_model.pkl' and 'label_encoder.pkl'")