from common_classes.EmailMessage import EmailMessage
import email
import re


class EnronMessage(EmailMessage):

    def extract_headers_main(self):
        message = email.message_from_string(self.body)
        extracted_headers = dict()
        for key in message.keys():
            extracted_headers[key] = message[key]
        self.headers = extracted_headers
        self.body = message.get_payload()

    def extract_headers_response(self):
        pattern = (
            r"From:\s(?P<From>.*)\n(?P<Headers>[\s\S]*?)\nSubject:\s(?P<Subject>.*)"
        )
        match = re.search(pattern, self.body)

        if match:
            from_header = match.group("From")
            subject_header = match.group("Subject")
            headers_block = match.group("Headers")

            intermediate_headers = dict(
                re.findall(r"^(.*?):\s*(.*)$", headers_block, re.MULTILINE)
            )

            headers = {"From": from_header, "Subject": subject_header}

            headers.update(intermediate_headers)

            email_content = re.sub(pattern, "", self.body, count=1)

            self.headers = headers
            self.body = email_content

    def extract_headers(self):
        if self.is_main:
            self.extract_headers_main()
        else:
            self.extract_headers_response()
