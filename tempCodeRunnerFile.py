# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
df = pd.read_csv('train.csv')  # Make sure your train.csv is in the same directory as this script

# Step 2: Clean the data (remove any rows with missing text or labels)
df = df.dropna(subset=['text', 'label'])

# Step 3: Split data into features (X) and labels (y)
X = df['text']  # News articles
y = df['label']  # Labels: 0 = Real, 1 = Fake

# Step 4: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Convert the text data into numerical data using TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)  # Fit and transform the training data
X_test_tfidf = tfidf.transform(X_test)  # Transform the testing data

# Step 6: Train a logistic regression model
model = LogisticRegression(max_iter=1000)  # Logistic Regression model
model.fit(X_train_tfidf, y_train)  # Train the model

# Step 7: Evaluate the model's performance on the test data
y_pred = model.predict(X_test_tfidf)  # Make predictions on the test set
accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Step 8: Function to classify a new piece of news as real or fake
def predict_news(text, threshold=0.6):
    """
    Predict whether a news article is real or fake.
    :param text: Input news article (string)
    :param threshold: Probability threshold for classifying as fake (default: 0.6)
    :return: Classification result ("Real News" or "Fake News")
    """
    text_tfidf = tfidf.transform([text])  # Convert input text into TF-IDF format
    prob = model.predict_proba(text_tfidf)[0]  # Get prediction probabilities
    print(f"Prediction Probabilities: [Real: {prob[0]:.4f}, Fake: {prob[1]:.4f}]")
    
    # Classify based on the threshold
    if prob[1] > threshold:
        return "Fake News"
    else:
        return "Real News"

# Step 9: Allow the user to input a piece of news and predict its label
if __name__ == "__main__":
    # Prompt the user to input a news article
    news_input = input("Enter a news article to classify: ")
    
    # Get the prediction
    prediction = predict_news(news_input, threshold=0.8)  # Adjust threshold as needed
    print(f'The news is classified as: {prediction}')