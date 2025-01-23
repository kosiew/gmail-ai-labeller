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
MAX_CONTEXT = 2048
MAX_CHARACTERS = MAX_CONTEXT * 4 - 150
MAX_CHARACTERS = 4000  # Overwrite as demonstration

# ----- Top-level Helpers -----

def authenticate_gmail():
    """
    Authenticate to Gmail using OAuth.
    Stores/loads credentials in 'token.pickle'.
    """
    print("==> Starting authenticate_gmail")
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
    creds = None

    # Load stored credentials if available
    if os.path.exists("token.pickle"):
        print("==> Loading stored credentials from token.pickle")
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, authenticate again
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("==> Refreshing expired credentials")
            creds.refresh(Request())
        else:
            print("==> Authenticating new credentials")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        print("==> Saving new credentials to token.pickle")
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    print("==> Finished authenticate_gmail")
    return creds

def extract_bracketed_content(text):
    """
    Extract content within square brackets, split by commas,
    and return a list of trimmed items.
    """
    bracketed_content = re.findall(r"\[(.*?)\]", text, re.DOTALL)
    if not bracketed_content:
        return []
    items = [item.strip() for item in bracketed_content[0].split(",")]
    return items

# ----- New Class: EmailClassifier -----

class EmailClassifier:
    """
    Dedicated class for classifying email content using an LLM.
    """

    def __init__(self, model=GPT4ALL_MODEL, valid_labels=None):
        self.model = model
        self.valid_labels = valid_labels if valid_labels else LABELS

    def classify(self, content):
        """
        Uses an LLM (via the 'llm' CLI tool) to classify the email text.
        Returns a list of valid labels.
        """
        truncated_content = content[:MAX_CHARACTERS]
        try:
            print("==> Starting classify_email_with_llm")
            labels_joined = ",".join(self.valid_labels)
            prompt = (
                "Here are valid labels.\n"
                f"{labels_joined}\n"
                "Please only return the most applicable label in square brackets "
                "without explanation eg [label1].\n"
                f"for this content:\n{truncated_content}"
            )

            result = subprocess.run(
                ["llm", "--model", self.model, prompt],
                capture_output=True,
                text=True,
            )

            stdout = result.stdout.strip()
            print(f"==> LLM response: {stdout}")

            # Extract bracketed content
            classification = extract_bracketed_content(stdout)
            if not classification:
                print("==> No bracketed content found")
                classification = ["etc"]
            else:
                classification = classification[:1]  # only keep first bracket
                print(f"==> Parsed classification: {classification}")

            # Ensure classification is among our valid labels
            classification = [
                label.lower() for label in classification
                if label.lower() in [l.lower() for l in self.valid_labels]
            ]
            print(f"==> classification sanitized: {classification}")
            return classification or ["etc"]

        except Exception as e:
            print(f"==> Error in classify_email_with_llm: {e}")
            return ["etc"]  # fallback label if error

# ----- Class: EmailProcessor -----

class EmailProcessor:
    """
    Handles message-level operations:
      - Caching labels
      - Creating labels
      - Applying labels to messages
      - Extracting message subject & content
    """

    def __init__(self, service):
        self.service = service
        self.cached_labels = {}
        self.init_cached_labels()

    def init_cached_labels(self):
        """
        Fetch and cache Gmail labels.
        """
        print("==> Fetching and caching Gmail labels")
        gmail_labels = (
            self.service.users().labels().list(userId="me").execute().get("labels", [])
        )
        for label in gmail_labels:
            # Store label by lower-cased name for easy lookup
            self.cached_labels[label["name"].lower()] = label["id"]

    def create_new_label(self, label_name):
        """
        Create a new label in Gmail.
        """
        print(f"==> Creating new label: {label_name}")
        new_label = (
            self.service.users()
            .labels()
            .create(
                userId="me",
                body={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show",
                },
            )
            .execute()
        )
        label_id = new_label["id"]
        self.cached_labels[label_name.lower()] = label_id
        return label_id

    def apply_labels(self, msg_id, labels):
        """
        Apply Gmail labels to the given message. Creates new labels if needed.
        """
        print("==> Starting apply_labels")
        label_ids = []

        # Limit which labels we apply (e.g., only two shortest)
        labels = [label.lower() for label in labels if 0 < len(label) <= 20]
        labels = sorted(labels, key=lambda x: len(x))[:2]
        labels.append(LABEL_PROCESSED)  # always apply the "processed" label

        for label_name in labels:
            label_id = self.cached_labels.get(label_name)
            if not label_id:
                # If label doesn't exist yet, create it
                label_id = self.create_new_label(label_name)
            label_ids.append(label_id)

        print(f"==> Applying labels to message ID: {msg_id} with label IDs: {label_ids}")
        self.service.users().messages().modify(
            userId="me", id=msg_id, body={"addLabelIds": label_ids}
        ).execute()
        print("==> Finished apply_labels")

    def get_subject_and_content(self, msg_id):
        """
        Retrieve the full message data from Gmail, extracting subject + text.
        """
        msg_data = (
            self.service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        # Extract subject
        headers = msg_data.get("payload", {}).get("headers", [])
        subject = next(
            (header["value"] for header in headers if header["name"] == "Subject"),
            "(No Subject)",
        )

        # Extract body text
        full_content = self._get_email_content(msg_data)
        return subject, full_content

    def _get_email_content(self, msg_data):
        """
        Extract text/plain parts from the full message payload.
        Fallback to snippet if no text/plain found.
        """
        email_body = ""
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])

        if parts:
            # Multi-part message
            for part in parts:
                mime_type = part.get("mimeType")
                if mime_type == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
                        email_body += decoded_data
                        print(f"==> Decoded part of email body: {decoded_data[:100]}...")
        else:
            # Single-part message
            body_data = payload.get("body", {}).get("data")
            if body_data:
                decoded_data = base64.urlsafe_b64decode(body_data).decode("utf-8")
                email_body += decoded_data
                print(f"==> Decoded single-part email body: {decoded_data[:100]}...")

        snippet = msg_data.get("snippet", "")
        return email_body if email_body else snippet


# ----- Class: EmailFetcher -----

class EmailFetcher:
    """
    Responsible for:
      - Retrieving emails (via pagination)
      - Delegating content extraction & label application to EmailProcessor
      - Delegating classification to EmailClassifier
    """

    def __init__(self, service, processor: EmailProcessor, classifier: EmailClassifier):
        self.service = service
        self.processor = processor
        self.classifier = classifier
        self.tabs = ["CATEGORY_UPDATES"]  # Adjust as needed

    def fetch_emails(self):
        """
        Fetch ALL emails from configured tabs and process them.
        """
        print("==> Starting fetch_emails")
        for tab in self.tabs:
            print(f"==> Fetching emails from tab: {tab}")
            self._fetch_and_process_emails(tab)
        print("✅ Finished fetch_emails successfully!")

    def _fetch_and_process_emails(self, tab):
        next_page_token = None
        total_fetched = 0
        total_labelled = 0

        while True:
            messages, next_page_token = self._fetch_paginated_emails(tab, next_page_token)
            total_fetched += len(messages)
            if not messages:
                print(f"==> No messages found in {tab}.")
                break

            for msg in messages:
                self._process_message(msg["id"])
                total_labelled += 1

            if not next_page_token:
                break

        print(f"==> Finished fetching from tab: {tab}. "
              f"Total fetched: {total_fetched}, total labelled: {total_labelled}")

    def _fetch_paginated_emails(self, tab, page_token):
        """
        Fetch emails from a specific Gmail tab, excluding archived/processed/sent.
        """
        query = f"-label:ARCHIVE -label:{LABEL_PROCESSED} -older_than:{OLDER_THAN} -in:sent"
        results = self.service.users().messages().list(
            userId="me",
            labelIds=[tab],
            q=query,
            pageToken=page_token,
        ).execute()

        messages = results.get("messages", [])
        next_page_token = results.get("nextPageToken")
        print(f"==> Fetched {len(messages)} messages from tab: {tab}")
        return messages, next_page_token

    def _process_message(self, msg_id):
        """
        Process a single message:
          - Retrieve content (subject/body)
          - Classify via EmailClassifier
          - Apply labels via EmailProcessor
        """
        print(f"==> Processing message ID: {msg_id}")
        subject, full_content = self.processor.get_subject_and_content(msg_id)
        labels = self.classifier.classify(full_content)
        print(f"==> subject: {subject} - Classified labels: {labels}")

        self.processor.apply_labels(msg_id, labels)
        print(f"==> Applied labels to message ID: {msg_id}")


# ----- Main -----

def main():
    # 1. Authenticate
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)

    # 2. Create the EmailProcessor (for label management & content extraction)
    processor = EmailProcessor(service)

    # 3. Create the EmailClassifier (for classification logic)
    classifier = EmailClassifier(model=GPT4ALL_MODEL, valid_labels=LABELS)

    # 4. Create the EmailFetcher to orchestrate fetching & processing
    fetcher = EmailFetcher(service, processor, classifier)

    # 5. Fetch & process all messages
    fetcher.fetch_emails()


if __name__ == "__main__":
    main()
