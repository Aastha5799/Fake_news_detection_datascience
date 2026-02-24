import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pickle

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
df = pd.read_csv('train.csv')
print("Dataset Loaded Successfully")

# -------------------------------
# STEP 2: Data Cleaning
# -------------------------------
# Remove missing values FIRST (important for performance)
df = df.dropna(subset=['text', 'label'])

# Convert text column to string
df['text'] = df['text'].astype(str)

print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['label'].value_counts())

# -------------------------------
# STEP 3: Visualize Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x=df['label'], hue=df['label'], palette='coolwarm', legend=False)
plt.title("Class Distribution: Real vs Fake News")
plt.xlabel("Label (0 = Real, 1 = Fake)")
plt.ylabel("Count")
plt.show()
plt.close()

# -------------------------------
# STEP 4: Train-Test Split
# -------------------------------
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 5: TF-IDF Vectorization
# Optimized to avoid slow training
# -------------------------------
tfidf = TfidfVectorizer(
    max_features=8000,        # Reduced for speed
    stop_words="english",
    ngram_range=(1,1)         # Unigram is enough for project
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF Vectorization Completed")

# -------------------------------
# STEP 6: Train Model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("Model Training Completed")

# -------------------------------
# STEP 7: Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# -------------------------------
# STEP 8: Confusion Matrix
# -------------------------------
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
plt.close()

# -------------------------------
# STEP 9: ROC Curve
# -------------------------------
y_prob = model.predict_proba(X_test_tfidf)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
plt.close()

# -------------------------------
# STEP 10: Save Model
# -------------------------------
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

print("\nModel and Vectorizer Saved Successfully!")

# -------------------------------
# STEP 11: Prediction Function
# -------------------------------
def predict_news(text, threshold=0.6):
    text_tfidf = tfidf.transform([text])
    prob = model.predict_proba(text_tfidf)[0]

    print(f"\nPrediction Probabilities:")
    print(f"Real: {prob[0]:.4f}, Fake: {prob[1]:.4f}")

    if prob[1] > threshold:
        return "Fake News"
    else:
        return "Real News"

# -------------------------------
# STEP 12: User Input
# -------------------------------
if __name__ == "__main__":
    news_input = input("\nEnter a news article to classify:\n")
    prediction = predict_news(news_input, threshold=0.8)
    print(f"\nPrediction: {prediction}")