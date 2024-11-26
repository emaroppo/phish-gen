import smtplib
import random
from constants import (
    EMAIL_ADDRESS,
    PHISHING_CAMPAIGN_DB,
    TEMPLATES_COLLECTION,
    TITLES_COLLECTION,
)
from data.QueryManager import query_manager
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from campaign.classes.TemplateEntry import TemplateEntry


class EmailSender:
    def __init__(self, sender_email, sender_password):
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.server = smtplib.SMTP("localhost", 1025)
        # self.server.starttls()
        # self.server.login(self.sender_email, self.sender_password)

    def send_email(self, to: str, template: TemplateEntry):
        # TO DO: fill placeholder
        msg = MIMEMultipart()
        msg["From"] = self.sender_email
        msg["To"] = to
        msg["Subject"] = template.title.title
        text = template.body
        msg.attach(MIMEText(text, "plain"))
        self.server.sendmail(self.sender_email, to, msg=msg.as_string())

    # quit server at the end
    def __del__(self):
        self.server.quit()
        print("Server quit")


email_sender = EmailSender(sender_email="example@local.com", sender_password="password")
