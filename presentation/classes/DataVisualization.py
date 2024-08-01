import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pydantic import BaseModel
from typing import Optional


class DataVisualization(BaseModel):
    out_dir: Optional[str] = "presentation/visualizations"

    def confusion_matrix(self, checkpoint, task):
        if not checkpoint.messages:
            checkpoint.get_messages()

        y_prompt = []
        y_output = []

        for message in checkpoint.messages:
            y_prompt.append(message.prompt[task])
            y_output.append(message.output[task])

        cm = confusion_matrix(
            y_prompt, y_output
        )  # y_prompt is on the y-axis, y_output is on the x-axis
        sns.heatmap(cm, annot=True, fmt="d")
        # annotate figure with axis labels, task and label names
        plt.title(f"Confusion Matrix for {task}")

        plt.xlabel("Output")
        plt.ylabel("Prompt")

        # add logic to add labels to the axes

        plt.show()

    def task_accuracy_over_steps(self, finetuned_model):
        # plot line graph of task accuracy over steps
        tasks_accuracy = dict()
        tasks_accuracy["urls"] = list()
        tasks_accuracy["attachments"] = list()
        tasks_accuracy["sentiment"] = list()

        for checkpoint in finetuned_model.checkpoints:
            tasks_accuracy["urls"].append(checkpoint.url_metrics["accuracy"])
            tasks_accuracy["attachments"].append(
                checkpoint.attachment_metrics["accuracy"]
            )
            tasks_accuracy["sentiment"].append(checkpoint.sentiment_metrics["accuracy"])

        steps = [checkpoint.steps for checkpoint in finetuned_model.checkpoints]

        # create dataframe from tasks_accuracy and steps
        tasks_accuracy_df = pd.DataFrame(tasks_accuracy, index=steps)
        # sort the dataframe by steps
        tasks_accuracy_df = tasks_accuracy_df.sort_index()

        plt.plot(tasks_accuracy_df)
        plt.legend(tasks_accuracy_df.columns)
        plt.xlabel("Steps")
        plt.ylabel("Accuracy")
        plt.title("Task Accuracy Over Steps")
        plt.show()

        # Saving logic
        pass

    def checkpoint_performance(self, checkpoint):
        # plot bar graph of task accuracy
        attachment_metrics = checkpoint.attachment_metrics
        sentiment_metrics = checkpoint.sentiment_metrics
        url_metrics = checkpoint.url_metrics

        metrics = {
            "attachments": attachment_metrics["accuracy"],
            "sentiment": sentiment_metrics["accuracy"],
            "urls": url_metrics["accuracy"],
        }

        plt.bar(metrics.keys(), metrics.values())
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.title("Task Accuracy")
        plt.show()

    def save(self, fig, name):
        fig.savefig(f"{self.out_dir}/{name}.png")
        pass
