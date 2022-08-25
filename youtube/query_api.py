# -*- coding: utf-8 -*-

# Sample Python code for youtube.playlistItems.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
import json

scopes = ["https://www.googleapis.com/auth/youtube.readonly"]
RESULTS_PER_PAGE = 50

def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "YOUR_CLIENT_SECRET_FILE.json"

    # Get credentials and create an API client
    flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
        client_secrets_file, scopes)
    credentials = flow.run_console()
    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials)

    all_titles = []
    nextPageToken = None
    
    while True:
        request = youtube.playlistItems().list(
            part="snippet",
            maxResults=50,
            pageToken = nextPageToken,
            playlistId="UUn8zNIfYAQNdrFRrr8oibKw"
        )
        response = request.execute()
        try:
            nextPageToken = response['nextPageToken']

            for item in response['items']:
                title = item['snippet']['title']
                all_titles.append(title)
                print(title)

        except:
            print(len(response['items']))
            break

    with open("data/titles.txt", "w") as f:
            for title in all_titles:
                f.write(f"{title}\n")

if __name__ == "__main__":
    main()


# Auth Code