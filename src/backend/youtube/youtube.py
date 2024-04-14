# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from googleapiclient.discovery import build
import csv
from googleapiclient.errors import HttpError
import pdb
import praw

app = Flask(__name__)
CORS(app)

# API key generated from the Google Cloud Console
API_KEY = "AIzaSyBVagAsTRg8e_Tu7AW5F8O36DuLIRoAFk0"

# YouTube API version
YOUTUBE_API_VERSION = "v3"

# YouTube code
def get_channel_comments():
    # comments = ["Comment 1", "Comment 2", "Comment 3"]
    youtube = build('youtube', YOUTUBE_API_VERSION, developerKey=API_KEY)

    #Extract video IDs from the channel
    channel_videos = youtube.search().list(
        part="id",
        channelId='UCXIJgqnII2ZOINSWNOGFThA',
        maxResults=10,  #Number of videos to Extract
    ).execute()

    #Extracting the video id from response we got in channel_videos
    video_ids = [item['id']['videoId'] for item in channel_videos['items'] if item['id']['kind'] == 'youtube#video']

    comments = []

    #Extracting comments for each video
    for video_id in video_ids:
        try:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=1,  #Number of comments to retrieve from each video
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

    return comments

@app.route('/get_comments', methods=['GET'])

def get_comments():
    comments = get_channel_comments()
    return jsonify(comments)

# Reddit code
def get_reddit_posts():
    #Assign Reddit instance
    reddit = praw.Reddit(client_id='vAOZJm-_U5096VaEjHTKoA',  #Client id
                     client_secret='wtTgPnvRy5GjqLUWbU1jr7qVeWzw9w',  #Client secret code
                     user_agent='Fake News Detector by Euphoric-Squash-3883') #Something unique that define your project

    #Mention the subreddit you want to extract posts from ex. 'politics', 'sports', 'worldnews'
    subreddit = reddit.subreddit('worldnews')

    #Extract hot posts from the subreddit (you can change 'hot' to 'new', 'top')
    posts = subreddit.top(limit=1)  #Number of the posts

    return posts

@app.route('/get_posts', methods=['GET'])

def get_posts():
    posts = get_reddit_posts()
    
    # Writing each post title as a new row
    post_titles = [post.title for post in posts]

    return jsonify(post_titles)

if __name__ == '__main__':
    app.run(debug=True)
