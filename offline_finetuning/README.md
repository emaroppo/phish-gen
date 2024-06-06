# Offline Finetuning
## Overview
The objective of this module is to produce a model that can generate a phishing email starting from a semi-structured prompt, such as a dictionary or JSON string. To achieve this, we fine-tune a model on a dataset of publicly available emails.
We store each email thread in a MongoDB collection, and parse the raw strings to extract the body and metadata of each email. We then preprocess the data, replacing URLs, phone numbers, dates, email addresses, attachments and signatures with placeholders. The actual values are stored as new fields of the message entry in the database. 
The metadata of each message is used to create a semistructured prompt, training the model to associate the email features with the email body.
