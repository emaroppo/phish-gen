# Offline Fine Tuning - Data Preprocessing

The pipeline for data preprocessing consists of 3 main classes: `DataImporter`, `EmailThread` and `EmailMessage`. 
The `DataImporter` class is responsible for reading the raw data from the source and storing it in a MongoDB collection.
The `EmailThread` class parses the raw strings, corresponding to an email thread and separates them into individual messages. It then stores the messages in a MongoDB collection, along with the metadata of each message. Broadly speaking, the `EmailThread` is responsible for handling the relationships between the messages in the email thread.
The `EmailMessage` class is used to parse the raw strings and extract the body and metadata from each message, including URLs, attachments etc..

Since the format of data is not consistent across the different sources, the pipeline for each source is constituted by subclasses of each of the 3 main classes, allowing for more flexibility.