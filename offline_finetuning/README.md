# Offline Finetuning
Train a model to generate an email from a semi-strucutred prompt describing desired features.

1. **Data Cleaning**:
    1. Import thread from files, store them in a database as strings
    2. Split threads into individual messages
    3. Extract headers from each message
2. **Data Preprocessing**: 
    1. Manually extract entities that follow patterns with low variability (e.g. phone numbers, dates, urls, attachments) using regular expressions, returning the value, type and position of the entity detected in the text
    2. Automatically extract entities with more unpredictable names using a Named Entity Recognition (NER) model
    3. Perform topic modelling
    4. Update database entries to include newly extracted features 
	
3. **Export**: 
    1. Retrieve relevant features for each message in the database
    2. Select and apply a prompt template, constructing a set of prompt-target pairs 
    3. Save and export the dataset as a PyTorch dataset (as well as other useful formats, e.g. for data validation)

4. **Model Training**:
    1. Load the dataset
    2. Train a (q)Lora model on the dataset
    3. Save the model and the tokenizer