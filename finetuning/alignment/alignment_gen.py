import streamlit as st
from inference.MessageGenerator import MessageGenerator
from finetuning.sft.classes.finetuned_models.FinetunedLLama31 import FinetunedLLama31
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["dpo_dataset"]


@st.cache_resource
def load_message_generator():
    finetuned_model = FinetunedLLama31.from_db(timestamp=1730656372)
    return MessageGenerator(finetuned_model=finetuned_model, checkpoint=2000)


message_generator = load_message_generator()

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
# press button to generate a message pair
button = st.button("Generate Message")
if button:
    message = message_generator.generate_message(**prompt)
    st.session_state.generated_message = message

if "generated_message" in st.session_state:
    message = st.session_state.generated_message
    st.write("Generated Message:")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Message")
        st.write(message)

    with col2:
        st.write("Revised Message")
        edited_message = st.text_area("Edit Message", value=message)
        if st.button("Select Revision", key="select_revision"):
            st.session_state.selected_revision = edited_message
            st.session_state.original_message = message

if "selected_revision" in st.session_state:
    db["dpo_dataset"].insert_one(
        {
            "chosen": st.session_state.selected_revision,
            "rejected": st.session_state.original_message,
        }
    )
    st.write("Inserted into database")
    del st.session_state.selected_revision
    del st.session_state.original_message
    del st.session_state.generated_message

# press neither button to generate a new message pair
st.button("Neither")
