import streamlit as st

from inference.MessageGenerator import MessageGenerator
from finetuning.sft.classes.FinetunedModel import FinetunedModel
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["dpo_dataset"]


finetuned_model = FinetunedModel.from_db(
    base_model_id="google/gemma-2b", timestamp=1727175863
)

message_generator = MessageGenerator(finetuned_model=finetuned_model, checkpoint=1500)

st.title("DPO Generator")
st.write(
    "This tool is used to generate data pairs for Direct Preference Optimization (DPO)."
)
col1, col2 = st.columns(2)
with col1:
    urls = st.radio("Include URLs", (True, False))
with col2:
    attachments = st.radio("Include Attachments", (True, False))
sentiment = st.multiselect(
    "Sentiment", ["neutral", "joy", "sadness", "fear", "anger", "surprise", "disgust"]
)
subject = st.text_input("Subject")

prompt = {
    "subject": subject,
    "urls": urls,
    "attachments": attachments,
    "sentiment": sentiment,
}
# press button to generate a messge pair
button = st.button("Generate Message Pair")
if button:
    message1 = message_generator.generate_message(**prompt)
    message2 = message_generator.generate_message(**prompt)

    st.write("Generated Message Pair:")
    col1, col2 = st.columns(2)
    # press button underneath better message to save the message pair in the dataset with appropriate labels
    with col1:
        st.write("Message 1")
        st.write(message1)
        button_1 = st.button("Select Message 1")
        if button_1:
            db["dpo_dataset"].insert_one(
                {"prompt": prompt, "chosen": message1, "rejected": message2}
            )

    with col2:
        st.write("Message 2")
        st.write(message2)
        button_2 = st.button("Select Message 2")
        if button_2:
            db["dpo_dataset"].insert_one({"chosen": message2, "rejected": message1})

    # press neither button to generate a new message pair
    st.button("Neither")
