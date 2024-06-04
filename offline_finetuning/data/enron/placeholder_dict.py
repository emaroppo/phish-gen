placeholder_dict = dict()
urls_dict = {
    "placeholder": "<URL>",
    "regex": [
        r"(https?://\S+)",
    ],
}
attachments_dict = {
    "placeholder": "<ATTACHMENT>",
    "regex": [
        r"<<File:(.*?)>>",
        r"\(See attached file:(.*?)\)",
        r"- ((?:.*?)\.(?:\w{1,4}))",
    ],
}

placeholder_dict["urls"] = urls_dict
placeholder_dict["attachments"] = attachments_dict
