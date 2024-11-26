import smtpd
import asyncore


class DebuggingServer(smtpd.SMTPServer):
    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        print(f"Receiving message from: {peer}")
        print(f"Message addressed from: {mailfrom}")
        print(f"Message addressed to: {rcpttos}")
        print(f"Message length: {len(data)}")
        print(f"Message data:\n{data}")
        return


if __name__ == "__main__":
    server = DebuggingServer(("localhost", 1025), None)
    print("SMTP Debugging Server started on localhost:1025")
    try:
        asyncore.loop()
    except KeyboardInterrupt:
        print("SMTP Debugging Server stopped")
