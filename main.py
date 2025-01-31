import os
import pickle
import subprocess
import base64
import re
import csv
from enum import Enum
from typing import (
    List,
    Tuple,
    Protocol,
    runtime_checkable,
    Iterator,
    Dict,
    Any,
    Optional,
)
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google.auth.exceptions import RefreshError

# For training (pandas, sklearn) – make sure you install them:
#   pip install pandas scikit-learn
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import typer
from nltk.stem import WordNetLemmatizer

DOMAIN_REGEX = re.compile(r'@([\w.-]+)')

app = typer.Typer()

# ----- Constants -----
GPT4ALL_MODEL = "mistral-7b-instruct-v0"
LABELS = ["programming", "news", "machine_learning", "etc"]
LABEL_PROCESSED = "processed"
LABEL_ARCHIVED = "ARCHIVE"
TOKEN_PICKLE = "token.pickle"
OLDER_THAN = "30d"
MAX_CONTEXT = 2048
MAX_CHARACTERS = MAX_CONTEXT * 4 - 150
MAX_CHARACTERS = 4000
MODEL_FILE="sklearn_email_model.pkl"

DEFAULT_QUERY_FILTER = (
    f"-label:{LABEL_ARCHIVED} -label:{LABEL_PROCESSED} -older_than:{OLDER_THAN} -in:sent"
)

EXTRACTED_EMAILS_CSV = "extracted_emails.csv"

# -------------------------------------------------------------------------
#                           Authentication Helper
# -------------------------------------------------------------------------


def authenticate_gmail():
    """
    Authenticate to Gmail using OAuth.
    Stores/loads credentials in TOKEN_PICKLE.
    """
    print("==> Starting authenticate_gmail")
    SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
    creds = None

    # Load stored credentials if available
    if os.path.exists(TOKEN_PICKLE):
        print(f"==> Loading stored credentials from {TOKEN_PICKLE}")
        with open(TOKEN_PICKLE, "rb") as token:
            creds = pickle.load(token)

    # If no valid credentials, authenticate again
    try:
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                print("==> Refreshing expired credentials")
                creds.refresh(Request())
            else:
                print("==> Authenticating new credentials")
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
                creds = flow.run_local_server(port=0)

            # Save credentials for future use
            print(f"==> Saving new credentials to {TOKEN_PICKLE}")
            with open(TOKEN_PICKLE, "wb") as token:
                pickle.dump(creds, token)
    except RefreshError:
        print("==> Refresh token has been expired or revoked. Re-authenticating.")
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

        # Save the new credentials for future use
        print(f"==> Saving new credentials to {TOKEN_PICKLE}")
        with open(TOKEN_PICKLE, "wb") as token:
            pickle.dump(creds, token)

    print("==> Finished authenticate_gmail")
    return creds


def get_gmail_service() -> build:
    """
    Authenticate and return the Gmail API service.
    """
    print("==> Authenticating to get Gmail API service")
    creds = authenticate_gmail()
    service = build("gmail", "v1", credentials=creds)
    return service


# -------------------------------------------------------------------------
#                           Email Data Class
# -------------------------------------------------------------------------
class EmailData:
    """
    Container for email data.
    """

    def __init__(
        self,
        subject: Optional[str] = None,
        full_content: Optional[str] = None,
        from_: Optional[str] = None,
    ):
        self.subject = subject
        self.full_content = full_content
        self.from_ = from_


class Folders(Enum):
    PROMOTIONS = "CATEGORY_PROMOTIONS"
    UPDATES = "CATEGORY_UPDATES"
    TRASH = "TRASH"
    NONE = ""

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

    def classify(self, content: str) -> List[str]: ...


@runtime_checkable
class IEmailProcessor(Protocol):
    """
    Protocol for processing individual emails:
      - extracting subject/body
      - applying labels
    """

    def apply_labels(self, msg_id: str, labels: List[str]) -> None:
        """
        Apply given labels to the specified message.
        """
        ...

    def get_email_data(self, msg_id: str) -> EmailData:
        """
        Return the EmailData object of an email.
        """
        ...


@runtime_checkable
class IEmailFetcher(Protocol):
    """
    Protocol for fetching emails in pages,
    delegating classification & processing to other protocols.
    """

    def fetch_emails(self) -> Iterator[Dict[str, Any]]:
        """
        Main entry point to fetch and process emails.
        """
        ...


# -------------------------------------------------------------------------
#                Concrete Implementations
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


class SklearnEmailClassifier(IEmailClassifier):
    """
    A scikit-learn-based classifier that loads a model file (pickle)
    and predicts labels from email content.
    """

    def __init__(self, model_path=MODEL_FILE):
        print(f"==> Loading sklearn model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def classify(self, content: str) -> List[str]:
        # Build a DataFrame with the columns that the pipeline expects.
        df = pd.DataFrame([{"From": "", "Subject": content}])
        prediction = self.model.predict(df)[0]
        return [prediction]


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

        print(
            f"==> Applying labels to message ID: {msg_id} with label IDs: {label_ids}"
        )
        self.service.users().messages().modify(
            userId="me", id=msg_id, body={"addLabelIds": label_ids}
        ).execute()
        print("==> Finished apply_labels")

    def get_subject_and_content(self, msg_id: str) -> Tuple[str, str]:
        """
        Retrieve the (subject, content) for a message.
        """
        email_data = self.get_email_data(msg_id)
        return email_data.subject, email_data.full_content

    def get_email_data(
        self, msg_id: str, fields: List[str] = ["from_", "subject", "content"]
    ) -> EmailData:
        """
        Retrieve the EmailData object for a message.
        Only fetches the specified fields: 'from_', 'subject', 'content'.
        """

        msg_data = (
            self.service.users()
            .messages()
            .get(userId="me", id=msg_id, format="full")
            .execute()
        )

        headers = msg_data.get("payload", {}).get("headers", [])
        subject = None
        from_ = None
        full_content = None

        if "subject" in fields:
            subject = next(
                (header["value"] for header in headers if header["name"] == "Subject"),
                "(No Subject)",
            )

        if "from_" in fields:
            from_ = next(
                (header["value"] for header in headers if header["name"] == "From"),
                "(No Sender)",
            )

        if "content" in fields:
            full_content = self._get_email_content(msg_data)

        return EmailData(subject=subject, full_content=full_content, from_=from_)

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
                        print(
                            f"==> Decoded part of email body: {decoded_data[:100]}..."
                        )
        else:
            body_data = payload.get("body", {}).get("data")
            if body_data:
                decoded_data = base64.urlsafe_b64decode(body_data).decode("utf-8")
                email_body += decoded_data
                print(f"==> Decoded single-part email body: {decoded_data[:100]}...")

        snippet = msg_data.get("snippet", "")
        return email_body if email_body else snippet


class EmailFetcher:
    """
    Class for fetching emails via pagination.
    """

    def __init__(
        self,
        service,
        tabs: List[Folders] = [Folders.UPDATES, Folders.PROMOTIONS],
        query_filter: str = DEFAULT_QUERY_FILTER,
    ):
        self.service = service
        self.tabs = tabs if tabs else [Folders.NONE]
        self.query_filter = query_filter

    def fetch_emails(self) -> Iterator[Dict[str, Any]]:
        """
        Fetch ALL emails from configured tabs.
        Yields one message at a time.
        """
        print("==> Starting fetch_emails")
        print(f"==> Query filter: {self.query_filter}")
        for tab in self.tabs:
            print(f"==> Fetching emails from tab: {tab.value}")
            next_page_token = None
            while True:
                messages, next_page_token = self._fetch_paginated_emails(
                    tab.value, next_page_token
                )
                for msg in messages:
                    yield msg

                if not next_page_token:
                    break
        print("✅ Finished fetch_emails successfully!")

    def _fetch_paginated_emails(self, tab: str, page_token: str):
        """
        Fetch emails from a specific Gmail tab,
        excluding archived/processed/sent messages.
        """
        list_args = {
            "userId": "me",
            "q": self.query_filter,
            "pageToken": page_token,
        }
        if tab:
            list_args["labelIds"] = [tab]

        results = self.service.users().messages().list(**list_args).execute()

        messages = results.get("messages", [])
        next_page_token = results.get("nextPageToken")
        print(f"==> Fetched {len(messages)} messages from tab: {tab}")
        return messages, next_page_token


class QueryFilterBuilder:
    def __init__(self):
        self.filters = []

    def add_filter(self, label: str, value: str):
        """Adds a filter condition to the query."""
        self.filters.append(f"{label}:{value}")
        return self
    
    def add_processed_filter(self, processed: bool):
        """Adds a filter condition to the query."""
        if processed:
            self.filters.append(f"label:{LABEL_PROCESSED}")
        else:
            self.filters.append(f"-label:{LABEL_PROCESSED}")
        return self

    def build(self):
        """Constructs the final query filter string."""
        unique_filters = list(set(self.filters))
        return " ".join(unique_filters)


class EmailLabeller:
    """
    Class for processing individual emails:
      - extracting subject/body
      - applying labels
    """

    def __init__(self, processor: IEmailProcessor, classifier: IEmailClassifier):
        self.processor = processor
        self.classifier = classifier

    def process_message(self, msg_id: str) -> None:
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




def label(classifier: IEmailClassifier):
    service = get_gmail_service()

    # Create concrete implementations for our protocols
    processor = DefaultEmailProcessor(service)
    labeller = EmailLabeller(processor, classifier)
    query_builder = QueryFilterBuilder()
    query_filter = (
        query_builder.add_filter("-in", "sent")
        .add_filter("-older_than", OLDER_THAN)
        .add_filter("-label", LABEL_ARCHIVED)
        .add_filter("-label", LABEL_PROCESSED)
        .build()
    )
    fetcher = EmailFetcher(service, query_filter=query_filter)
    for email in fetcher.fetch_emails():
        msg_id = email["id"]
        labeller.process_message(msg_id)


# -------------------------------------------------------------------------
#                          Typer Commands
# -------------------------------------------------------------------------
@app.command()
def append_to_extracted_emails(
    input_csv: str = EXTRACTED_EMAILS_CSV, output_csv: str = f"_{EXTRACTED_EMAILS_CSV}"
) -> None:
    """
    Append rows from input_csv to output_csv.
    If output_csv is blank, include the header line (From, Subject, Label).
    Otherwise, only append the data rows.
    """
    print(f"==> Appending rows from {input_csv} to {output_csv}")

    # Read input CSV
    with open(input_csv, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Check if output CSV is blank
    try:
        with open(output_csv, "r", newline="", encoding="utf-8") as outfile:
            existing_rows = list(csv.reader(outfile))
    except FileNotFoundError:
        existing_rows = []

    # Append rows to output CSV
    with open(output_csv, "a", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        if not existing_rows:
            # Include header if output CSV is blank
            writer.writerow(["From", "Subject", "Label"])
        # Append data rows
        writer.writerows(rows[1:])  # Skip header row from input CSV

    print(f"==> Appended rows to {output_csv}")


@app.command()
def llm_label():
    classifier = DefaultEmailClassifier(model=GPT4ALL_MODEL, valid_labels=LABELS)
    label(classifier)


@app.command()
def sklearn_label():
    classifier = SklearnEmailClassifier(model_path=MODEL_FILE)
    label(classifier)


@app.command()
def extract_data_from_emails(
    folder: Folders = Folders.UPDATES.value,
    output_csv: str = EXTRACTED_EMAILS_CSV,
    processed: bool = False,
) -> None:
    """
    Fetches all emails with or without the 'processed' label,
    extracts the 'From' and 'Subject',
    and writes them to a CSV file so you can manually add labels for training.
    """

    print("==> Starting extract_data_from_emails")
    processor, fetcher = create_email_fetcher(folder, processed)

    # 3. Write results to CSV
    write_emails_to_csv(output_csv, processor, fetcher)

@app.command()
def extract_data_from_trash(
    output_csv: str = EXTRACTED_EMAILS_CSV,
    processed: bool = False,
) -> None:
    """
    Fetches all emails with or without the 'processed' label,
    extracts the 'From' and 'Subject',
    and writes them to a CSV file so you can manually add labels for training.
    """

    print("==> Starting extract_data_from_emails")
    processor, fetcher = create_trash_email_fetcher(processed)

    # 3. Write results to CSV
    write_emails_to_csv(output_csv, processor, fetcher)

@app.command()
def extract_data_with_filter(
    gmail_filter: str,
    output_csv: str = EXTRACTED_EMAILS_CSV,
    processed: bool = False,
) -> None:
    """
    Fetches all emails matching the given Gmail filter,
    extracts the 'From' and 'Subject',
    and writes them to a CSV file so you can manually add labels for training.
    example gmail_filter: "from:pycoder older_than:60d -older_than:220d in:anywhere"
    """

    print(f"==> Starting extract_data_with_filter with filter: {gmail_filter}")
    processor, fetcher = create_gmail_filter_fetcher(gmail_filter, processed)

    # Write results to CSV
    write_emails_to_csv(output_csv, processor, fetcher)

def create_gmail_filter_fetcher(gmail_filter: str, processed: bool):
    service = get_gmail_service()

    tabs = []
    processor = DefaultEmailProcessor(service)
    query_builder = QueryFilterBuilder()
    filters = gmail_filter.split()
    for filter_item in filters:
        key, value = filter_item.split(":", 1)
        query_builder.add_filter(key, value)
    query_builder.add_processed_filter(processed)

    if not tabs:
        query_builder.add_filter("in", "anywhere")

    query_filter = query_builder.build()
    fetcher = EmailFetcher(service, tabs=tabs, query_filter=query_filter)
    return processor,fetcher

def create_email_fetcher(folder, processed):
    service = get_gmail_service()

    processor = DefaultEmailProcessor(service)
    query_builder = QueryFilterBuilder()
    query_builder.add_filter("-in", "sent").add_filter(
        "-older_than", OLDER_THAN
    ).add_filter("-label", LABEL_ARCHIVED)
    query_builder.add_processed_filter(processed)

    query_filter = query_builder.build()
    fetcher = EmailFetcher(service, tabs=[folder], query_filter=query_filter)
    return processor,fetcher

def create_trash_email_fetcher(processed: bool):
    gmail_filter = f"in:trash -older_than:{OLDER_THAN}"
        
    return create_gmail_filter_fetcher(gmail_filter, processed)

def write_emails_to_csv(output_csv, processor, fetcher):
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["From", "Subject", "Label"])  # label is empty initially

        seen = set()
        counter = 0
        for msg in fetcher.fetch_emails():
            msg_id = msg["id"]
            # Get minimal metadata for "From" and "Subject" only

            # Usage example
            email_data = processor.get_email_data(msg_id, fields=["from_", "subject"])
            from_, subject = email_data.from_, email_data.subject

            if (from_, subject) not in seen:
                writer.writerow([from_, subject, ""])
                seen.add((from_, subject))
                print(f"==> Extracted: {from_} - {subject}")
                counter += 1

    print(f"==> Extracted {counter} emails into {output_csv}")



class EmailDomainExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.extract_domain(str(email)) for email in X]
    
    @staticmethod
    def extract_domain(email):
        match = DOMAIN_REGEX.search(email)
        return match.group(1).lower() if match else "unknown"


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.clean_text(str(text)) for text in X]
    
    @staticmethod
    def clean_text(text):
        lemmatizer = WordNetLemmatizer()
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)  # Remove special characters
        return ' '.join(
            lemmatizer.lemmatize(word) 
            for word in text.split() 
            if word not in ENGLISH_STOP_WORDS
        )

@app.command()
def train_sklearn_model_from_csv(
    input_csv: str = "_extracted_emails.csv",
    model_path: str = MODEL_FILE,
) -> None:

    print("==> Starting train_sklearn_model_from_csv")

    # 1. Load data
    df = pd.read_csv(input_csv)

    # 2. Drop rows with missing or empty labels
    df = df.dropna(subset=["Label"])
    df["Label"] = df["Label"].astype(str)
    df = df[df["Label"].str.strip() != ""]

    # 3. Extract features/target
    X = df[["From", "Subject"]]
    y = df["Label"].astype(str)

    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Build a ColumnTransformer pipeline
    preprocessor = ColumnTransformer([
        ("subject_pipeline",
         Pipeline([
             ("clean_text", TextCleaner()),
             ("tfidf_subject", TfidfVectorizer()),
         ]),
         "Subject"),
        ("domain_pipeline",
         Pipeline([
             ("extract_domain", EmailDomainExtractor()),
             ("tfidf_domain", TfidfVectorizer()),
         ]),
         "From"),
        ("from_tfidf",
         TfidfVectorizer(),
         "From"),
    ])

    # 6. Full pipeline with classifier
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", MultinomialNB())
    ])

    # (Optional) Hyperparameter tuning with GridSearch
    param_grid = {
        "preprocessor__subject_pipeline__tfidf_subject__ngram_range": [(1,1), (1,2)],
        "clf__alpha": [0.01, 0.1, 1.0, 5.0, 10.0]    
    }
    grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro', cv=5)

    # 7. Train
    grid_search.fit(X_train, y_train)

    # 8. Evaluate
    y_pred = grid_search.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 9. Save model
    with open(model_path, "wb") as f:
        pickle.dump(grid_search, f)

    print(f"==> Trained model (GridSearchCV) saved to '{model_path}'")


if __name__ == "__main__":
    app()
