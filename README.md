# Phishing Email Generator
[![Optional Image Description](https://github.com/emaroppo/phish-gen/blob/main/docs/diagrams/thumbs/Dissertation_Diagram_thumb.png?raw=true)](https://github.com/emaroppo/phish-gen/blob/main/docs/diagrams/Dissertation_Diagram.png?raw=true)
## Overview
The aim of this project is to deliver an effective generator of phishing emails. The project has 4 (possible) components:
1. An offline finetuning module, which will be used to produce a model that can generate phishing emails based on the prompt given by the user.
2. A message generator class, which takes the the output of the model (the email body), injects it into an appropriate email template (e.g. font, styles, html etc.), replaces placeholders with the desired values (e.g. a malicious url) and sends the email to the target(s). 
3. An evaluation and feedback module, which will be used gather data and evaluate the effectiveness of the generated phishing emails.
4. An online finetuning module, which will be used to finetune the model based on the feedback gathered from the evaluation and feedback module.
5. A RAG module, which will be used to inject relevant information into the phishing email generated by the model.
- TO DO: Review and define scope

## 1. Offline Finetuning
The goal of this module is to produce a model that can generate a phishing email starting from a semi-strucutred prompt, such as a dictionary or JSON string. To achieve this, the module handles several tasks:
1. **Data Cleaning**: The dataset is a collection of leaked emails collated from various sources. For this reason, the format of the emails is not consistent. Furthermore, each file corresponds to an email thread, meaning it may contain several messages, as well as metadata such as "sender", "subject" etc.. The DataImporter classes (one per source) read the files composing the raw dataset, extract a string corresponding to the individual threads and stores them in a MongoDB collection. The EmailThread and EmailMessage classes are then used to parse the raw strings and extract body and metadata, storing each in its own field.
2. **Data Preprocessing**: We wish the model to only learn the general content of the emails, not specific details such as dates and phone numbers. In fact, we may want to programmatically inject such details into the generated emails. For this reason, URLs, phone numbers, dates, email addresses, attachments and signatures are replaced with placeholders. The actual values are stored as new fields of the message entry in the database. Additional fields may be added in this phase using models of the BERT family (e.g. BERTopic) to label messages on the basis of their content. All changes are saved in the database.
3. **Dataset Export**: A simple projection is retrieved for each message in the dataset. The export method takes as input a list of prompt fields and a list of target fields. The dataset will then consist of a list of stringified dictionaries, where each dictionary contains the prompt and target fields (and their respective values).
3. **Model Training**: The dataset is used to train a LoRA from one of the various open source models available. Before starting the training, the placeholders added in step 2 are added to the vocabulary of the model. This is done to ensure that the model can gain an understanding of the placeholders and learn to generate them in the right context. The model is then trained on the dataset, and the weights are saved to disk.

- Done:
    1. *EnronDataImporter* correctly imports the data from the Enron dataset.
    2. *EnronEmailThread* and EnronEmailMessage mostly separate the email threads and messages correctly, but there are still some mistakes; base logic to inject placeholders is implemented, but still needs to be tested and the regular expressions still need to be defined
    3. *EnronEmailMessage* correctly extracts the body and metadata from the email messages, but there are still some mistakes
    4. *DatasetFactory* correctly creates a dataset from the database for the decoder model, still need to test the method to create the dataset for the models used for labelling in prefix finetuning
    5. LoRA training script should work, but is still untested on the dataset and does not yet include the placeholders in the vocabulary

- In progress:
    1. *DNCDataImporter* is still in progress; a significant number of the threads are stored as html files, which need to be parsed correctly

- To Do:
    1. Define the placeholders to be used in the dataset and test the preprocessing step
    2. Define models to be used for the additional fields and label dataset and test *ModelLabeller* class
    3. Train LoRa to parse/clean the dataset
    4. Define prompt and target fields for the dataset export
    5. Containerize mongodb for dataset portability

## 2. Message Generator
This class is pretty straightforward. It has a *.generate_message()* method, which takes a prompt and passes is to the model. It then takes the output of the model, injects it into a style template and replaces the values of the placeholders with the desired values.

## 3. Evaluation and Feedback
The goal of this module is to gather data on the effectiveness of the generated phishing emails. Much still needs to be defined, but the goal would be to inject the emails with some tracking mechanisms to gather data on the user's interaction with the email. In first instance, this could be compared to the effectiveness of a phishing email generated by a human. The data gathered will then be used to finetune the model in the next module. Alternatively, a method of evaluating the emails without user interaction (or with a limited of users) needs to be established.

## 4. Online Finetuning
The feedback gathered by step 3 can then be used to further finetune the model using RLHF. An evaluator model will be trained on the feedback data, and the model will be finetuned using the evaluator model (coupled with a penalty function for divergence) as reward.

## 5. RAG Module
TBD. Added it to the diagram mostly as a reminder that it is a possibility. In broad strokes, the idea would be to scrape the web for relevant information on the target (e.g. using social media, company websites etc.) and use this information to more effectively persuade or impersonate the target.