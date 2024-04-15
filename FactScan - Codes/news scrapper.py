import requests
from bs4 import BeautifulSoup

# Define the target URL
url = "http://factchecker.in/"

# Get the HTML content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all elements containing news articles (replace with appropriate selectors for the website)
articles = soup.find_all('div', class_='story')  # This might need adjustment

# Initialize empty lists to store article data
article_titles = []
article_content = []

# Extract article title and content (adjust based on website structure)
for article in articles:
    headline = article.find('h2').text.strip()  # Assuming title is in an h2 tag
    content_block = article.find('p', class_='summary')  # Assuming summary is in a paragraph with class 'summary'
    if content_block:  # Check if content block exists
        content = content_block.text.strip()
    else:
        content = ""  # Handle cases where there's no summary element
    article_titles.append(headline)
    article_content.append(content)

# Print the first 5 scraped titles and content (for demonstration purposes)

if len(articles) > 0:  # Check if there are any articles
    for i in range(len(articles)):
        # Access elements within the loop only if there are enough articles
        print(f"Title: {article_titles[i]}")
        print(f"Content: {article_content[i]}\n")
else:
    print("No articles found on this page.")

