from data.processing.labellers.MessageLabeller import MessageLabeller


class SentimentMessageLabeller(MessageLabeller):

    def label_message(self, message_body, batch_size=1):
        labels = self.classifier(
            message_body,
            max_length=512,
            truncation=True,
            padding="max_length",
            batch_size=batch_size,
        )
        if type(message_body) == str:
            return labels[0]
        return labels

    def validate_label(self, labels: list, message_body: str):
        message_label = self.label_message(message_body)

        top_n = len(labels)
        # retrieve the top_n labels with the highest scores
        top_labels = sorted(message_label, key=lambda x: x["score"], reverse=True)[
            :top_n
        ]
        top_labels = [label["label"] for label in top_labels]
        # check if the labels match the top_n labels
        return all(label in top_labels for label in labels)
