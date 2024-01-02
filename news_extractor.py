import os
from newsapi import NewsApiClient
from newspaper import Article
from datetime import datetime, timedelta

def extract_articles(api_key, category, from_date, to_date, output_folder):
    # Initialize NewsApiClient with your API key
    newsapi = NewsApiClient(api_key=api_key)

    # Convert date strings to datetime objects
    from_date = datetime.strptime(from_date, '%Y-%m-%d')
    to_date = datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1)  # Add one day to include articles from the specified end date

    # Define parameters for the API request
    articles = newsapi.get_everything(
        q=f'{category}',
        language='en',
        from_param=from_date.isoformat(),
        to=to_date.isoformat(),
        sort_by='relevancy',
        page_size=100,
        page=1,
    )

    # Check if the API request was successful
    if articles['status'] == 'ok':
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Extract articles and create text files
        for index, article_info in enumerate(articles['articles']):
            title = article_info['title']
            url = article_info['url']

            # Fetch the full article content using newspaper3k
            article = Article(url)
            article.download()
            article.parse()
            content = article.text

            # Create a unique filename based on the article title
            filename = f"{output_folder}/{category}_{index + 1}.txt"

            # Write article content to a text file
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(f"Title: {title}\n\n")
                file.write(f"Content: {content}\n")

            print(f"Article {index + 1} saved to {filename}")

    else:
        print(f"Error: {articles['message']}")

# Replace 'YOUR_API_KEY' with your actual Google News API key
api_key = '966f516f6f754ba4937b9f193f3ff06a'

# Customize the parameters
category = 'inflation'
from_date = '2023-11-20'
to_date = '2023-11-30'
output_folder = 'output_articles'

# Call the function to extract articles
extract_articles(api_key, category, from_date, to_date, output_folder)
