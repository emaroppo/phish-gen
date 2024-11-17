import streamlit as st

from inference.MessageGenerator import MessageGenerator
from finetuning.sft.classes.finetuned_models.FinetunedModel import FinetunedModel
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["kto_dataset"]


finetuned_model = FinetunedModel.from_db(
    base_model_id="google/gemma-2b", timestamp=1723627438
)

message_generator = MessageGenerator(finetuned_model=finetuned_model, checkpoint=2500)

st.title("KTO Generator")
st.write(
    "This tool is used to generate data pairs for Kahneman-Tversky Optimization (KTO)."
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
button = st.button("Generate Message")
if button:
    message = message_generator.generate_message(**prompt)

    st.write("Generated Message:")
    # press button underneath better message to save the message pair in the dataset with appropriate labels
    with col1:
        st.write("Message")
        st.write(message)
        button_1 = st.button("+")
        button_2 = st.button("-")
        if button_1:
            db["kto_dataset"].insert_one(
                {"prompt": prompt, "message": message, "label": 1}
            )
        if button_2:
            db["kto_dataset"].insert_one(
                {"prompt": prompt, "message": message, "label": 0}
            )
