from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from googleapiclient.errors import HttpError
from googleapiclient.discovery import build
from flask import Flask, request, jsonify
from imblearn.over_sampling import SMOTE
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect
from flask_cors import CORS
import pandas as pd
import joblib
import string
import pickle
import nltk
import praw
import re


app = Flask(__name__)
CORS(app)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize the WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# API key generated from the Google Cloud Console
API_KEY = "AIzaSyBVagAsTRg8e_Tu7AW5F8O36DuLIRoAFk0"

# YouTube API version
YOUTUBE_API_VERSION = "v3"

previous_comments_data = []

# YouTube code
@app.route('/get_comments', methods=['GET'])

def get_comments():
    youtube = build('youtube', YOUTUBE_API_VERSION, developerKey=API_KEY)
    comments = []

    # BCC News Channel: UC16niRr50-MSBwiO3YDb3RA
    # FOX News Channel: UCXIJgqnII2ZOINSWNOGFThA
    channel_ids = ['UCXIJgqnII2ZOINSWNOGFThA', 'UC16niRr50-MSBwiO3YDb3RA']

    for channel_id in channel_ids:

        #Extract video IDs from the channel
        channel_videos = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=1000,  #Number of videos to Extract
        ).execute()

        #Extracting the video id from response we got in channel_videos
        video_ids = [item['id']['videoId'] for item in channel_videos['items'] if item['id']['kind'] == 'youtube#video']

        #Extracting comments for each video
        for video_id in video_ids:
            try:
                response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=1000,  #Number of comments to retrieve from each video
                    textFormat="plainText"
                ).execute()

                # print("response :", response)

                for item in response["items"]:
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comments.append(comment)
            #Handle the error of videos with disbaled comments 
            except HttpError as e:
                if e.resp.status == 403:
                    print("Comments are disabled for video ID:" + video_id + " Skipping.")
                else:
                    raise  #Raise the exception again if it's not a 403 error

    return jsonify(comments)

# Reddit code
@app.route('/get_posts', methods=['GET'])

def get_posts():
    #Assign Reddit instance
    reddit = praw.Reddit(client_id='vAOZJm-_U5096VaEjHTKoA',  #Client id
                     client_secret='wtTgPnvRy5GjqLUWbU1jr7qVeWzw9w',  #Client secret code
                     user_agent='Fake News Detector by Euphoric-Squash-3883') #Something unique that define your project

    #Mention the subreddit you want to extract posts from ex. 'politics', 'sports', 'worldnews'
    subreddit = reddit.subreddit('worldnews')

    #Extract hot posts from the subreddit (you can change 'hot' to 'new', 'top')
    posts = subreddit.top(limit=1)  #Number of the posts
    
    # Writing each post title as a new row
    post_titles = [post.title for post in posts]

    return jsonify(post_titles)

# Function to remove punctuation, stopwords, and non-English comments, and perform lemmatization
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords and perform lemmatization
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]

    # Join the tokens back into a single string
    clean_text = ' '.join(clean_tokens)

    return clean_text

# Function to detect English text
def is_english(text):
    try:
        return detect(text) == 'en'
    except Exception as e:  # This will catch errors like detecting language of empty strings or strings with no linguistic content
        return False

# Function to load models
def load_models():
    # Load the model and vectorizer
    model = joblib.load('rf_classifier.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return model, vectorizer

# Function to predict comment
def predict_comment(comment, model, vectorizer):
    # Preprocess the comment
    comment = comment.lower()
    comment = re.sub(r"[^a-zA-Z\s]", '', comment)

    # Vectorize the comment
    comment_vectorized = vectorizer.transform([comment])

    # Make a prediction
    prediction = model.predict(comment_vectorized)

    # Return the predicted class
    return 'Cyberbullying' if prediction[0] == 1 else 'Non-Cyberbullying'

# Cyberbullying Check code
@app.route('/check_comment', methods=['POST'])

def check_comment():
    global previous_comments_data
    data = request.json
    print("data",data['inputValue'])
    # print("data",data['commentsData'])

    if data['commentsData'] != previous_comments_data:
        print("in if")
        previous_comments_data = data['commentsData']

        # Create a DataFrame with the comments_array data
        df = pd.DataFrame(data['commentsData'], columns=['Comment'])

        # Clean the retrieved comments
        df['Processed_Comment'] = df['Comment'].map(clean_text)

        # Load the vectorizer
        with open("/Users/apple/Desktop/CapstoneProject/sample/src/backend/api_service_call/vectorizer.pkl", "rb") as file:
            vectorizer = pickle.load(file)

        # Transform the processed comments using the loaded vectorizer
        X_new = vectorizer.transform(df['Processed_Comment'])

        # Load the trained model
        with open("/Users/apple/Desktop/CapstoneProject/sample/src/backend/api_service_call/LinearSVC.pkl", "rb") as file:
            model = pickle.load(file)

        # Use the loaded model to make predictions on the new data
        predictions = model.predict(X_new)

        # Add the predictions to your DataFrame
        df['Predictions'] = predictions

        print(df)

        # Ensure that the 'Comment' column is retained
        if 'Comment' not in df.columns:
            df['Comment'] = data['commentsData']

        # Separate the minority and majority classes
        df_minority = df[df['Predictions'] == 1]
        df_majority = df[df['Predictions'] == 0]

        # Oversample minority class
        df_minority_oversampled = df_minority.sample(len(df_majority), replace=True, random_state=42)

        # Combine the oversampled minority class with the majority class
        df_balanced = pd.concat([df_majority, df_minority_oversampled])

        # Shuffle the dataset to avoid any ordering bias
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        # Check if there are more than one class in the predictions
        if len(df['Predictions'].unique()) > 1:
            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_new, predictions)

            # Combine the resampled data into a DataFrame
            df_balanced['Predictions'] = y_resampled

            # Shuffle the resampled DataFrame
            df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

            # Save the resampled DataFrame to a CSV file
            df_balanced.to_csv('balanced_dataset.csv', index=False)

            print(df_balanced)

            # Load the balanced dataset
            # dataset = pd.read_csv('/Users/apple/Desktop/CapstoneProject/sample/src/backend/cyberbullying/balanced_dataset.csv')

            # Remove non-English rows
            df_balanced['is_english'] = df_balanced['Comment'].apply(is_english)
            dataset = df_balanced[df_balanced['is_english']]

            # Clean text data
            dataset.loc[:, 'Comment'] = dataset['Comment'].str.replace(r"[^a-zA-Z\s]", '', regex=True)
            dataset.loc[:, 'Comment'] = dataset['Comment'].str.lower()

            # Shuffle the dataset
            dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

            # Initialize vectorizer
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

            # Split dataset into features and target variable
            X = dataset['Comment']
            y = dataset['Predictions']

            # Fit and transform the vectorizer on the entire dataset
            X_tfidf = vectorizer.fit_transform(X)

            # Initialize Random Forest Classifier
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

            # Train the classifier on the entire dataset
            rf_classifier.fit(X_tfidf, y)

            # Save the model
            joblib.dump(rf_classifier, 'rf_classifier.joblib')

            # Save the vectorizer
            joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

            # Load the trained model and vectorizer
            model, vectorizer = load_models()

            # Get the prediction
            result = predict_comment(data['inputValue'], model, vectorizer)

            # Print the result
            print(f"The comment is classified as: {result}")
        else:
            # Handle the case where there is only one class in predictions
            return jsonify({'error': 'The target variable has only one class. Unable to apply SMOTE.'})
    else:
        # Load the trained model and vectorizer
        model, vectorizer = load_models()

        # Get the prediction
        result = predict_comment(data['inputValue'], model, vectorizer)

        # Print the result
        print(f"The comment is classified as: {result}")

    return jsonify(result)

# Function to predict whether the input news is real or fake
def predict_news(input_news):
    # Load the labeled DataFrame containing clustered posts
    clustered_df = pd.read_csv(r'/Users/apple/Desktop/CapstoneProject/sample/src/backend/api_service_call/clustered_labeled_posts.csv')

    # Split the data into features (text) and labels
    X = clustered_df['Post']
    y = clustered_df['Label']

    # Text preprocessing and feature extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X)

    # Train a Random Forest classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X, y)

    # Save the trained classifier and vectorizer for later use
    joblib.dump(vectorizer, 'vectorizer.joblib')
    joblib.dump(classifier, 'classifier.joblib')

    # Load the trained classifier and vectorizer
    vectorizer = joblib.load('vectorizer.joblib')
    classifier = joblib.load('classifier.joblib')

    # Preprocess the input news
    input_news_vectorized = vectorizer.transform([input_news])
    # Predict using the classifier
    prediction = classifier.predict(input_news_vectorized)

    return prediction[0]

# FakeNews Check code
@app.route('/check_post', methods=['POST'])

def check_news():   
    data = request.json  # Capture new input data inside the loop
    input_value = data.get('inputValue', '').lower()

    prediction = predict_news(input_value)
    print("Prediction:", prediction)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(debug=True)


