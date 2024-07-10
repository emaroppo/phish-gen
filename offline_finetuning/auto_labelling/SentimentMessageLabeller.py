from offline_finetuning.auto_labelling.MessageLabeller import MessageLabeller


class SentimentMessageLabeller(MessageLabeller):

    def label_message(self, message_body: str):
        labels = self.classifier(
            message_body, max_length=512, truncation=True, padding="max_length"
        )[0]

        return labels
