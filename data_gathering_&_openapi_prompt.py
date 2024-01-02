import openai
from newsapi import NewsApiClient
import json



# Init
newsapi = NewsApiClient(api_key='966f516f6f754ba4937b9f193f3ff06a')
openai.api_key = "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"

# /v2/top-headlines
all_articles = newsapi.get_everything(q='commodities',
                                      sources='bbc-news',
                                      domains='bbc.co.uk',
                                      from_param='2023-09-27',
                                      to='2023-10-02',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)

print(all_articles)

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a financial analyst and have to say if the news are positive or negative in regards of the news you read."},
    {"role": "system", "content": "Answer by saying call if the news is positive and short if it is negative."},
    {"role": "user", "content": "Nasdaq Futures Sink as Earnings Misses Punished: Markets Wrap"}
  ]
)

print(completion.choices[0].message)