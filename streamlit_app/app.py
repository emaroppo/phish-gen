import streamlit as st
from streamlit_option_menu import option_menu
import json
import plotly.express as px
import pandas as pd

# options: dataset, model
with open("streamlit_app/summary.json") as f:
    dataset_data = json.load(f)
    timestamps = [dataset["timestamp"] for dataset in dataset_data]

with open("streamlit_app/model_summary.json") as f:
    model_data = json.load(f)
    model_timestamps = [model["timestamp"] for model in model_data]

sidebar = st.sidebar
# insert option menu in column 1
with sidebar:
    option = option_menu("Select", ["Dataset", "Model"])
    if option == "Dataset":
        st.header(f"{option} Metrics")
        dataset_timestamp = st.selectbox("Select Dataset", timestamps)

    elif option == "Model":
        st.header(f"{option} Metrics")
        model_timestamp = st.selectbox("Select Model", model_timestamps)

        model_data = next(
            model for model in model_data if model["timestamp"] == model_timestamp
        )
        model_checkpoints = [
            checkpoint["steps"] for checkpoint in model_data["checkpoints"]
        ]
        model_checkpoints.sort()
        model_checkpoint = st.selectbox(
            "Select Checkpoint", ["Overview"] + model_checkpoints
        )


st.header(f"{option} Metrics")


if option == "Dataset":
    dataset_data = next(
        dataset for dataset in dataset_data if dataset["timestamp"] == dataset_timestamp
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Dataset {dataset_data['timestamp']}")
        # Percentage of sentiments pie chart with plotly
        st.write("Sentiments")
        sentiments = dataset_data["sentiments"]
        sentiments_df = pd.DataFrame(columns=["Sentiment", "Percentage"])
        for key, value in sentiments.items():
            sentiments_df = sentiments_df.append(
                {"Sentiment": key, "Percentage": value}, ignore_index=True
            )

        fig = px.pie(sentiments_df, values="Percentage", names="Sentiment")

        st.plotly_chart(fig)

    with col2:
        st.write(f"Message Count: {dataset_data['message_count']}")
        # Attachment formats pie chart with plotly
        st.write("Attachments")

        attachments = dataset_data["attachment_formats"]
        attachments_df = pd.DataFrame(columns=["Attachment", "Count"])
        for key, value in attachments.items():
            attachments_df = attachments_df.append(
                {"Attachment": key, "Count": value}, ignore_index=True
            )
        fig = px.pie(attachments_df, values="Count", names="Attachment")
        st.plotly_chart(fig)

    st.write("Entities")
    # plot entity counts as separate bars on a chart
    entities = dataset_data["entity_counts"]
    entities_df = pd.DataFrame(columns=["Entity", "Count"])

    for key, value in entities.items():
        entities_df = entities_df.append(
            {"Entity": key, "Count": value}, ignore_index=True
        )
    fig = px.bar(entities_df, x="Entity", y="Count")
    st.plotly_chart(fig)

elif option == "Model":
    # select model using timestamp

    if model_checkpoint == "Overview":
        # plot model overview
        st.write("Model Overview")
        tasks_accuracy = dict()
        tasks_accuracy["urls"] = list()
        tasks_accuracy["attachments"] = list()
        tasks_accuracy["sentiment"] = list()

        for checkpoint in model_data["checkpoints"]:
            tasks_accuracy["urls"].append(checkpoint["url_metrics"]["accuracy"])
            tasks_accuracy["attachments"].append(
                checkpoint["attachment_metrics"]["accuracy"]
            )
            tasks_accuracy["sentiment"].append(
                checkpoint["sentiment_metrics"]["accuracy"]
            )

        steps = [checkpoint["steps"] for checkpoint in model_data["checkpoints"]]

        # create dataframe from tasks_accuracy and steps, plot as line graph
        tasks_accuracy_df = pd.DataFrame(tasks_accuracy, index=steps)
        # sort the dataframe by steps
        tasks_accuracy_df = tasks_accuracy_df.sort_index()

        fig = px.line(tasks_accuracy_df)
        st.plotly_chart(fig)
    else:
        # plot checkpoint performance
        checkpoint_data = next(
            checkpoint
            for checkpoint in model_data["checkpoints"]
            if checkpoint["steps"] == model_checkpoint
        )

        st.write(f"Checkpoint {model_checkpoint}")

        metrics = {
            "attachments": checkpoint_data["attachment_metrics"]["accuracy"],
            "sentiment": checkpoint_data["sentiment_metrics"]["accuracy"],
            "urls": checkpoint_data["url_metrics"]["accuracy"],
        }
        metrics_df = pd.DataFrame(columns=["Task", "Accuracy"])
        for key, value in metrics.items():
            metrics_df = metrics_df.append(
                {"Task": key, "Accuracy": value}, ignore_index=True
            )

        fig = px.bar(metrics_df, x="Task", y="Accuracy")
        st.plotly_chart(fig)
