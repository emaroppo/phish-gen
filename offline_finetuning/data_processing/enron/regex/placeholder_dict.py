placeholder_dict = dict()
urls_dict = {
    "placeholder": "URL",
    "regex": [
        r"(https?:\/\/\S+)",
        r"(www\.\S+(?:\.\w+)+\b)",
    ],
}
attachments_dict = {
    "placeholder": "ATTACHMENT",
    "regex": [
        r"<<\s+File:(.*?)>>",
        r"<<File\s+Attachment:\s+(.*?)>>",
        r"\(See attached file:(.*?)\)",
        r"<<((?:.*?)\.(?:[A-z0-9]{1,4}))>>",
        r"- ((?:.*?)\.(?:[A-z0-9]{1,4}))",
    ],
}

month_regex = r"(?:(?:0?\d)|(?:1[0-3]))"
day_regex = r"(?:(?:[0-3]?\d)|(?:3[01]))"
year_regex = r"(?:((19)|(20))?\d{2})"  # year; accept 19xx and 20xx

dates_dict = {
    "placeholder": "DATE",
    "regex": [
        r"((?:(?:0?\d)|(?:1[0-3]))[\.\-/ ](?:(?:[0-3]?\d)|(?:3[01]))[\.\-/ ](?:(?:(?:19)|(?:20))?\d{2}))",  # month, day, year
        r"((?:(?:[0-3]?\d)|(?:3[01]))[\.\-/ ](?:(?:0?\d)|(?:1[0-3]))[\.\-/ ](?:(?:(?:19)|(?:20))?\d{2}))",  # day, month, year
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}(?:th|st|nd|rd)?,? \d{4})",  # month, day, year
        r"(\d{1,2}(?:th|st|nd|rd)? (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{4})",  # day, month, year
        r"(\d{1,2}(?:th|st|nd|rd)? (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.? \d{4})",  # day, month, year
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.? \d{1,2}(?:th|st|nd|rd)?,? \d{4})",  # month, day, year
    ],
}

phone_numbers_dict = {
    "placeholder": "PHONE",
    "regex": [
        r"((?:\+\d{1,2} )?\d{3}-\d{3}-\d{4})",
        r"((?:\+\d{1,2} )?\(\d{3}\) \d{3}-\d{4})",
        r"((?:\+\d{1,2} )?\d{3}\.\d{3}\.\d{4})",
        r"((?:\+\d{1,2} )?\d{3} \d{3} \d{4})",
    ],
}

placeholder_emails_dict = {
    "placeholder": "EMAIL",
    "regex": [
        r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*)@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
    ],
}

placeholder_dict["emails"] = placeholder_emails_dict
placeholder_dict["urls"] = urls_dict
placeholder_dict["attachments"] = attachments_dict
placeholder_dict["dates"] = dates_dict
placeholder_dict["phone_numbers"] = phone_numbers_dict
