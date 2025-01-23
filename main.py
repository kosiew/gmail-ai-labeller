import os
import pickle
import subprocess
import base64
import re

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ----- Constants -----
GPT4ALL_MODEL = "mistral-7b-instruct-v0"
LABELS = ["programming", "news", "machine_learning", "etc"]
LABEL_PROCESSED = "processed"
OLDER_THAN = "30d"
MAX_CHARACTERS = 4000

# ----- Authentication -----

def authenticate_gmail():
    """Authenticate to Gmail using OAuth."""
    print("==> Authenticating Gmail")
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return creds

# ----- Interfaces -----

class IEmailProcessor:
    def apply_labels(self, msg_id, labels):
        pass
    
    def get_subject_and_content(self, msg_id):
        pass

class IEmailClassifier:
    def classify(self, content):
        pass

# ----- Implementations -----

class EmailProcessor(IEmailProcessor):
    """Handles Gmail operations: extracting content & applying labels."""
    def __init__(self, service):
        self.service = service
        self.cached_labels = {}
        self.init_cached_labels()

    def init_cached_labels(self):
        gmail_labels = self.service.users().labels().list(userId="me").execute().get("labels", [])
        for label in gmail_labels:
            self.cached_labels[label["name"].lower()] = label["id"]

    def create_new_label(self, label_name):
        new_label = self.service.users().labels().create(
            userId="me", body={"name": label_name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}
        ).execute()
        self.cached_labels[label_name.lower()] = new_label["id"]
        return new_label["id"]

    def apply_labels(self, msg_id, labels):
        label_ids = []
        labels.append(LABEL_PROCESSED)
        for label_name in labels:
            label_id = self.cached_labels.get(label_name.lower()) or self.create_new_label(label_name)
            label_ids.append(label_id)
        self.service.users().messages().modify(userId="me", id=msg_id, body={"addLabelIds": label_ids}).execute()

    def get_subject_and_content(self, msg_id):
        msg_data = self.service.users().messages().get(userId="me", id=msg_id, format="full").execute()
        subject = next((header["value"] for header in msg_data.get("payload", {}).get("headers", []) if header["name"] == "Subject"), "(No Subject)")
        content = self._get_email_content(msg_data)
        return subject, content

    def _get_email_content(self, msg_data):
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])
        if parts:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    data = part["body"].get("data")
                    return base64.urlsafe_b64decode(data).decode("utf-8") if data else ""
        return msg_data.get("snippet", "")

class EmailClassifier(IEmailClassifier):
    """Classifies emails using an LLM."""
    def __init__(self, model=GPT4ALL_MODEL, valid_labels=None):
        self.model = model
        self.valid_labels = valid_labels if valid_labels else LABELS

    def classify(self, content):
        truncated_content = content[:MAX_CHARACTERS]
        prompt = (
            f"Valid labels: {', '.join(self.valid_labels)}.\n"
            "Return the best label in square brackets (e.g., [label]).\n"
            f"Content: {truncated_content}"
        )
        result = subprocess.run(["llm", "--model", self.model, prompt], capture_output=True, text=True)
        labels = re.findall(r"\[(.*?)\]", result.stdout)
        return [label.lower() for label in labels if label.lower() in self.valid_labels] or ["etc"]

# ----- Fetcher & Handler -----

class EmailFetcher:
    """Fetches emails from Gmail."""
    def __init__(self, service):
        self.service = service

    def fetch_emails(self, query):
        results = self.service.users().messages().list(userId="me", q=query).execute()
        return results.get("messages", [])

class EmailHandler:
    """Orchestrates fetching, classification, and processing."""
    def __init__(self, fetcher, processor: IEmailProcessor, classifier: IEmailClassifier):
        self.fetcher = fetcher
        self.processor = processor
        self.classifier = classifier

    def process_emails(self):
        emails = self.fetcher.fetch_emails(f"-label:ARCHIVE -older_than:{OLDER_THAN}")
        for email in emails:
            subject, content = self.processor.get_subject_and_content(email["id"])
            labels = self.classifier.classify(content)
            self.processor.apply_labels(email["id"], labels)

# ----- Main Execution -----

def main():
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)

    fetcher = EmailFetcher(service)
    processor = EmailProcessor(service)
    classifier = EmailClassifier(model=GPT4ALL_MODEL, valid_labels=LABELS)
    handler = EmailHandler(fetcher, processor, classifier)
    
    handler.process_emails()

if __name__ == "__main__":
    main()
