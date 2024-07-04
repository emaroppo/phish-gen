from offline_finetuning.auto_labelling.MessageLabeller import MessageLabeller


class NERMessageLabeller(MessageLabeller):

    def label_message(self, message_body: str):
        labels = super().label_message(message_body)
        # compose the tokens back into word
        words = list()
        first_label = labels.pop(0)
        word_start = first_label["start"]
        word_end = first_label["end"]
        word_label = first_label["entity"].split("-")[1]

        for label in labels:

            if label["entity"][0] == "I":
                word_end = label["end"]
            elif label["entity"][0] == "B":
                words.append((word_start, word_end, word_label))
                word_start = label["start"]
                word_end = label["end"]
                word_label = label["entity"].split("-")[1]

        for word in words:
            print(message_body[word[0] : word[1]], word[2])

        return words
