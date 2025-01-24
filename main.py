import os
import pickle
import subprocess
import base64
import re
import csv

from typing import List, Tuple, Protocol, runtime_checkable, Iterator, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# For training (pandas, sklearn) – make sure you install them:
#   pip install pandas scikit-learn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import typer

app = typer.Typer()

# ----- Constants -----
GPT4ALL_MODEL = "mistral-7b-instruct-v0"
LABELS = ["programming", "news", "machine_learning", "etc"]
LABEL_PROCESSED = "processed"
OLDER_THAN = "30d"
MAX_CONTEXT = 2048
MAX_CHARACTERS = MAX_CONTEXT * 4 - 150
MAX_CHARACTERS = 4000

# -------------------------------------------------------------------------
#                           Authentication Helper
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
#                          Protocol Definitions
# -------------------------------------------------------------------------


@runtime_checkable
class IEmailClassifier(Protocol):
    """
    Protocol for an email classifier.
    Concrete implementations must provide a `classify` method 
    that returns a list of labels.
    """
    def classify(self, content: str) -> List[str]:
        ...


@runtime_checkable
class IEmailProcessor(Protocol):
    """
    Protocol for processing individual emails:
      - extracting subject/body
      - applying labels
    """
    def get_subject_and_content(self, msg_id: str) -> Tuple[str, str]:
        """
        Return the (subject, content) of an email.
        """
        ...

    def apply_labels(self, msg_id: str, labels: List[str]) -> None:
        """
        Apply given labels to the specified message.
        """
        ...


@runtime_checkable
class IEmailFetcher(Protocol):
    """
    Protocol for fetching emails in pages, 
    delegating classification & processing to other protocols.
    """
    def fetch_emails(self) -> None:
        """
        Main entry point to fetch and process emails.
        """
        ...

# -------------------------------------------------------------------------
#                Concrete Implementation: EmailClassifier
# -------------------------------------------------------------------------

def extract_bracketed_content(text: str) -> List[str]:
    """
    Helper: Extract content within square brackets, split by commas,
    and return a list of trimmed items.
    """
    bracketed_content = re.findall(r"\[(.*?)\]", text, re.DOTALL)
    if not bracketed_content:
        return []
    items = [item.strip() for item in bracketed_content[0].split(",")]
    return items

class DefaultEmailClassifier:
    """
    Default implementation of IEmailClassifier using a local LLM ('llm' CLI).
    Conforms to IEmailClassifier by virtue of having a matching `classify` method.
    """

    def __init__(self, model: str = GPT4ALL_MODEL, valid_labels: List[str] = None):
        self.model = model
        self.valid_labels = valid_labels if valid_labels else LABELS

    def classify(self, content: str) -> List[str]:
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

            classification = extract_bracketed_content(stdout)
            if not classification:
                print("==> No bracketed content found")
                classification = ["etc"]
            else:
                # Take only the first bracket
                classification = classification[:1]
                print(f"==> Parsed classification: {classification}")

            # Ensure classification is among our valid labels
            lower_valid_labels = [lbl.lower() for lbl in self.valid_labels]
            classification = [
                label.lower()
                for label in classification
                if label.lower() in lower_valid_labels
            ]
            print(f"==> classification sanitized: {classification}")
            return classification or ["etc"]

        except Exception as e:
            print(f"==> Error in classify_email_with_llm: {e}")
            return ["etc"]  # fallback on error

# -------------------------------------------------------------------------
#               Concrete Implementation: EmailProcessor
# -------------------------------------------------------------------------

class DefaultEmailProcessor:
    """
    Default implementation of IEmailProcessor:
      - Caches Gmail labels
      - Creates new labels if needed
      - Extracts subject/body from messages
      - Applies labels to messages
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
            self.cached_labels[label["name"].lower()] = label["id"]

    def create_new_label(self, label_name: str) -> str:
        """
        Create a new label in Gmail if it doesn't exist.
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

    def apply_labels(self, msg_id: str, labels: List[str]) -> None:
        """
        Apply Gmail labels to the given message. Creates new labels if needed.
        """
        print("==> Starting apply_labels")
        label_ids = []

        # Filter/limit labels if needed
        labels = [label.lower() for label in labels if 0 < len(label) <= 20]
        labels = sorted(labels, key=lambda x: len(x))[:2]  # e.g. keep 2 shortest
        labels.append(LABEL_PROCESSED)  # always apply "processed"

        for label_name in labels:
            label_id = self.cached_labels.get(label_name)
            if not label_id:
                label_id = self.create_new_label(label_name)
            label_ids.append(label_id)

        print(f"==> Applying labels to message ID: {msg_id} with label IDs: {label_ids}")
        self.service.users().messages().modify(
            userId="me", id=msg_id, body={"addLabelIds": label_ids}
        ).execute()
        print("==> Finished apply_labels")

    def get_subject_and_content(self, msg_id: str) -> Tuple[str, str]:
        """
        Retrieve the (subject, content) for a message.
        """
        msg_data = (
            self.service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        # Subject
        headers = msg_data.get("payload", {}).get("headers", [])
        subject = next(
            (header["value"] for header in headers if header["name"] == "Subject"),
            "(No Subject)",
        )

        # Body
        full_content = self._get_email_content(msg_data)
        return subject, full_content

    def _get_email_content(self, msg_data: dict) -> str:
        """
        Extract text/plain parts from message payload. Fallback to snippet if none found.
        """
        email_body = ""
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])

        if parts:
            for part in parts:
                mime_type = part.get("mimeType")
                if mime_type == "text/plain":
                    data = part["body"].get("data")
                    if data:
                        decoded_data = base64.urlsafe_b64decode(data).decode("utf-8")
                        email_body += decoded_data
                        print(f"==> Decoded part of email body: {decoded_data[:100]}...")
        else:
            body_data = payload.get("body", {}).get("data")
            if body_data:
                decoded_data = base64.urlsafe_b64decode(body_data).decode("utf-8")
                email_body += decoded_data
                print(f"==> Decoded single-part email body: {decoded_data[:100]}...")

        snippet = msg_data.get("snippet", "")
        return email_body if email_body else snippet

# -------------------------------------------------------------------------
#               Concrete Implementation: EmailFetcher
# -------------------------------------------------------------------------

class DefaultEmailFetcher:
    """
    Default implementation of IEmailFetcher:
      - Retrieves emails via pagination
      - Delegates content extraction to an IEmailProcessor
      - Delegates classification to an IEmailClassifier
      - Then instructs the processor to apply labels
    """

    def __init__(
        self,
        service,
        processor: IEmailProcessor,
        classifier: IEmailClassifier,
        tabs: List[str] = None
    ):
        self.service = service
        self.processor = processor
        self.classifier = classifier
        self.tabs = tabs or ["CATEGORY_UPDATES"]

    def fetch_emails(self) -> None:
        """
        Fetch ALL emails from configured tabs, process them, 
        applying classification & labels.
        """
        print("==> Starting fetch_emails")
        for tab in self.tabs:
            print(f"==> Fetching emails from tab: {tab}")
            self._fetch_and_process_emails(tab)
        print("✅ Finished fetch_emails successfully!")

    def _fetch_and_process_emails(self, tab: str) -> None:
        next_page_token = None
        total_fetched = 0
        total_processed = 0

        while True:
            messages, next_page_token = self._fetch_paginated_emails(tab, next_page_token)
            total_fetched += len(messages)

            if not messages:
                print(f"==> No messages found in {tab}.")
                break

            for msg in messages:
                self._process_message(msg["id"])
                total_processed += 1

            if not next_page_token:
                break

        print(f"==> Finished fetching from tab: {tab}. "
              f"Total fetched: {total_fetched}, total processed: {total_processed}")

    def _fetch_paginated_emails(self, tab: str, page_token: str):
        """
        Fetch emails from a specific Gmail tab, 
        excluding archived/processed/sent messages.
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

    def _process_message(self, msg_id: str) -> None:
        """
        Orchestrate processing a single message:
          - get subject/content (via IEmailProcessor)
          - classify (via IEmailClassifier)
          - apply labels (via IEmailProcessor)
        """
        print(f"==> Processing message ID: {msg_id}")
        subject, full_content = self.processor.get_subject_and_content(msg_id)

        labels = self.classifier.classify(full_content)
        print(f"==> subject: {subject} - Classified labels: {labels}")

        self.processor.apply_labels(msg_id, labels)
        print(f"==> Applied labels to message ID: {msg_id}")

# -------------------------------------------------------------------------
#               NEW: Extract Data From Processed Emails
# -------------------------------------------------------------------------

@app.command()
def extract_data_from_processed_emails(output_csv: str = "extracted_emails.csv") -> None:    
    """
    Fetches all emails with label 'processed',
    extracts the 'From' and 'Subject', 
    and writes them to a CSV file so you can manually add labels for training.
    """

    print("==> Starting extract_data_from_processed_emails")

    # 1. Authenticate
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)

    # 2. Collect all messages labeled 'processed'
    all_messages = []
    page_token = None

    while True:
        response = service.users().messages().list(
            userId="me",
            labelIds=[LABEL_PROCESSED],
            pageToken=page_token
        ).execute()

        messages = response.get("messages", [])
        if not messages:
            break

        all_messages.extend(messages)
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    # 3. Write results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["From", "Subject", "Label"])  # label is empty initially

        for msg in all_messages:
            msg_id = msg["id"]
            # Get minimal metadata for "From" and "Subject" only
            msg_data = service.users().messages().get(
                userId="me",
                id=msg_id,
                format="metadata",
                metadataHeaders=["From", "Subject"]
            ).execute()

            headers = msg_data.get("payload", {}).get("headers", [])
            from_ = next((h["value"] for h in headers if h["name"] == "From"), "")
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "")

            writer.writerow([from_, subject, ""])

    print(f"==> Extracted {len(all_messages)} 'processed' emails into {output_csv}")

# -------------------------------------------------------------------------
#               NEW: Train scikit-learn Model from CSV
# -------------------------------------------------------------------------
@app.command()
def train_sklearn_model_from_csv(
    input_csv: str = "extracted_emails.csv",
    model_path: str = "sklearn_email_model.pkl"
) -> None:
    """
    1) Expects a CSV with columns: "From", "Subject", "Label"
    2) Splits data into train/test
    3) Trains a TF-IDF + LogisticRegression pipeline
    4) Prints classification report
    5) Saves the model pipeline to a .pkl file
    """
    print("==> Starting train_sklearn_model_from_csv")

    # Load data
    df = pd.read_csv(input_csv)

    # We expect columns: From, Subject, Label
    # Drop rows with missing or empty labels
    df = df.dropna(subset=["Label"])
    df = df[df["Label"].str.strip() != ""]
    
    # Combine "From" + "Subject" for training features (simple approach)
    X = df["From"].astype(str) + " " + df["Subject"].astype(str)
    y = df["Label"].astype(str)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"==> Trained model saved to '{model_path}'")

# -------------------------------------------------------------------------
#               NEW: Sklearn-based Email Classifier
# -------------------------------------------------------------------------

class SklearnEmailClassifier(IEmailClassifier):
    """
    A scikit-learn-based classifier that loads a model file (pickle)
    and predicts labels from email content.
    """

    def __init__(self, model_path="sklearn_email_model.pkl"):
        print(f"==> Loading sklearn model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def classify(self, content: str) -> List[str]:
        # Make a prediction using the pipeline
        # You can, if you like, add additional preprocessing steps here
        prediction = self.model.predict([content])[0]
        # Return a list of labels (in this simple case, just one predicted label)
        return [prediction]


# -------------------------------------------------------------------------
#                                 Main
# -------------------------------------------------------------------------
def get_gmail_service() -> build:
    """
    Authenticate and return the Gmail API service.
    """
    print("==> Authenticating to get Gmail API service")
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)
    return service

@typer.command()
def llm_label():
    service = get_gmail_service()

    # 2. Create concrete implementations for our protocols
    processor = DefaultEmailClassifier(service)
    classifier = DefaultEmailClassifier(model=GPT4ALL_MODEL, valid_labels=LABELS)
    fetcher = DefaultEmailFetcher(
        service=service,
        processor=processor,
        classifier=classifier,
        tabs=["CATEGORY_UPDATES"]
    )

    # 3. Use the fetcher (which uses the processor & classifier) to fetch & process emails
    fetcher.fetch_emails()
    
# def main():
    # EXAMPLE usage:
    # 1. Classify emails with the *existing* DefaultEmailClassifier (LLM-based).
    #    (Uncomment if you want to run your normal classification flow.)

    # creds = authenticate_gmail()
    # service = build("gmail", "v1", credentials=creds)
    # processor = DefaultEmailProcessor(service)
    # classifier = DefaultEmailClassifier(model=GPT4ALL_MODEL, valid_labels=LABELS)
    # fetcher = DefaultEmailFetcher(
    #     service=service,
    #     processor=processor,
    #     classifier=classifier,
    #     tabs=["CATEGORY_UPDATES"]
    # )
    # fetcher.fetch_emails()

    # 2. Extract data from already processed emails to a CSV
    # extract_data_from_processed_emails(output_csv="extracted_emails.csv")

    # 3. After labeling the CSV manually, train a scikit-learn model:
    # train_sklearn_model_from_csv(
    #     input_csv="extracted_emails.csv", 
    #     model_path="sklearn_email_model.pkl"
    # )

    # 4. Now switch to the SklearnEmailClassifier:
    # creds = authenticate_gmail()
    # service = build("gmail", "v1", credentials=creds)
    # processor = DefaultEmailProcessor(service)
    # sklearn_classifier = SklearnEmailClassifier(model_path="sklearn_email_model.pkl")
    # fetcher_with_sklearn = DefaultEmailFetcher(
    #     service=service,
    #     processor=processor,
    #     classifier=sklearn_classifier,
    #     tabs=["CATEGORY_UPDATES"]
    # )
    # fetcher_with_sklearn.fetch_emails()

    # print("==> Done. Uncomment the desired workflow in main() to use.")
    

if __name__ == "__main__":
    app()
