from data.enron.EnronDataImporter import EnronDataImporter

enron = EnronDataImporter("data/enron/maildir", "enron_emails")
enron.load_raw_data()
enron.isolate_multiparts()
enron.clean_multiparts()
