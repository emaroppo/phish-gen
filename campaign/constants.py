import json

with open("inference/emails.json", "r") as f:
    EMAIL_ADDRESS = json.load(f)

PHISHING_CAMPAIGN_DB = "phishing_campaign"
TEMPLATES_COLLECTION = "templates"
MESSAGES_COLLECTION = "messages"
TITLES_COLLECTION = "titles"
