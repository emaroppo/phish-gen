from data.enron.EnronDataImporter import EnronDataImporter

enron = EnronDataImporter("data/enron/maildir", "enron_emails")
enron.split_multipart_data()
