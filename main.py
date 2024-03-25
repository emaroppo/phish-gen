from data.enron.EnronDataImporter import EnronDataImporter

enron = EnronDataImporter("data/enron/dataset/maildir", "enron_emails2")
enron.clean_multiparts()
