from data.enron.EnronDataImporter import EnronDataImporter

enron = EnronDataImporter("data/enron/dataset/maildir", "enron_emails_test")
enron.load_raw_data()
enron.isolate_multiparts()
enron.clean_multiparts()

