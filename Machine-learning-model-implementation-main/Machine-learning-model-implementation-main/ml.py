from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
emails = ["Win a free iPhone now!", "Important meeting at 10 AM", "Get rich fast!", "Lunch plans?", "Limited offer, click now!"]
labels = [1, 0, 1, 0, 1]  # 1: Spam, 0: Not Spam

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict & evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")