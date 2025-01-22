import os
import pickle
import subprocess
import base64
import base64

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

GPT4ALL_MODEL="orca-mini-3b-gguf2-q4_0"
LABELS=["programming", "news", "machine_learning", "etc"]
LABEL_PROCESSED="processed"
cached_labels = None

def authenticate_gmail():
    """
    Authenticate to Gmail using OAuth. 
    Stores/loads credentials in 'token.pickle'.
    """
    print("==> Starting authenticate_gmail")
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    
    # Load stored credentials if available
    if os.path.exists('token.pickle'):
        print("==> Loading stored credentials from token.pickle")
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate again
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("==> Refreshing expired credentials")
            creds.refresh(Request())
        else:
            print("==> Authenticating new credentials")
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        print("==> Saving new credentials to token.pickle")
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    print("==> Finished authenticate_gmail")
    return creds

def classify_email_with_llm(content):
    """
    Uses an LLM (via the 'llm' CLI tool) to classify the email text.
    If it's marketing or not related to Python/Rust/web dev => "Archive"
    Otherwise => "Do Not Archive"
    
    Returns a string: either "Archive" or "Do Not Archive".
    """
    labels = ",".join(LABELS)
    try:
        print("==> Starting classify_email_with_llm")
        prompt = (
            "Here are some labels.\n"
            f"{labels}\n"
            "Please return a list of applicable labels.\n"
            f"for this content:{content}"
        )
        print(f"==> Generated prompt: {prompt}")
        
        result = subprocess.run(
            ["llm", "complete", "--model", GPT4ALL_MODEL, prompt],
            capture_output=True,
            text=True
        )
        print(f"==> LLM response: {result.stdout.strip()}")

        classification = result.stdout.strip()
        # classification is labels delimited with comma, 
        # return a list of labels
        classification = classification.split(",")
        print(f"==> Parsed classification: {classification}")
        return classification

    except Exception as e:
        print(f"==> Error in classify_email_with_llm: {e}")
        # If an error occurs, default to "Do Not Archive"
        return ["error"]

def apply_labels(service, msg_id, labels):
    """
    Create (if needed) and apply Gmail labels to the given message.
    """
    print("==> Starting apply_labels")
    global cached_labels 
    if cached_labels is None:
        cached_labels = {}
        # Get existing labels
        gmail_labels = service.users().labels().list(userId='me').execute().get('labels', [])
        print(f"==> Fetched existing labels: {gmail_labels}")
        for label in gmail_labels:
            cached_labels[label['name']] = label['id']
    else:
        print("==> Using cached labels")

    label_ids = []
    
    for label_name in labels:
        label_id = cached_labels.get(label_name)
        print(f"==> Processing label: {label_name}")
        
        # If label does not exist, create a new label
        if not label_id:
            print(f"==> Creating new label: {label_name}")
            new_label = service.users().labels().create(
                userId='me',
                body={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show"
                }
            ).execute()
            label_id = new_label['id']
            cached_labels[label_name] = label_id
            print(f"==> Created new label: {label_name} with ID: {label_id}")
        
        label_ids.append(label_id)
    
    print(f"==> Applying labels to message ID: {msg_id} with label IDs: {label_ids}")
    # Apply all labels to the email at once
    service.users().messages().modify(
        userId='me',
        id=msg_id,
        body={'addLabelIds': label_ids}
    ).execute()
    print("==> Finished apply_labels")


def fetch_emails():
    """
    Fetch ALL emails from Gmail (Inbox + Tabs: Social, Promotions, Updates, Forums),
    classify them, and apply labels accordingly.
    Uses pagination to fetch ALL messages.
    """
    print("==> Starting fetch_emails")
    
    # Authenticate and build service
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    
    # Gmail tab labels
    TABS = ["INBOX", "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"]
    TABS = ["CATEGORY_UPDATES"]
    processed_label_id = cached_labels.get(LABEL_PROCESSED)
    
    for tab in TABS:
        print(f"==> Fetching emails from tab: {tab}")
        next_page_token = None  # Start pagination
        total_fetched = 0

        while True:
            # Fetch emails (paginate if needed)
            results = service.users().messages().list(
                userId='me', 
                labelIds=[tab], 
                pageToken=next_page_token
            ).execute()

            messages = results.get('messages', [])
            next_page_token = results.get('nextPageToken')  # Check for next page
            total_fetched += len(messages)

            print(f"==> Fetched {len(messages)} messages from tab: {tab}, Total: {total_fetched}")

            if not messages:
                print(f"==> No messages found in {tab}.")
                break  # Stop if no messages exist

            for msg in messages:
                print(f"==> Processing message ID: {msg['id']}")
                msg_data = service.users().messages().get(
                    userId='me', 
                    id=msg['id'], 
                    format='full'   # 'full' to get the entire message payload
                ).execute()
                
                # Check if the message already has the "processed" label
                msg_label_ids = msg_data.get('labelIds', [])
                
                if processed_label_id and processed_label_id in msg_label_ids:
                    print(f"==> Message ID: {msg['id']} already has the 'processed' label. Skipping.")
                    continue
                # Get the full text from the message payload
                email_body = ""
                payload = msg_data.get('payload', {})

                # If the payload has parts, iterate over them
                parts = payload.get('parts', [])
                if parts:
                    for part in parts:
                        mime_type = part.get('mimeType')
                        if mime_type == 'text/plain':
                            data = part['body'].get('data')
                            if data:
                                decoded_data = base64.urlsafe_b64decode(data).decode('utf-8')
                                email_body += decoded_data
                                print(f"==> Decoded part of email body: {decoded_data[:100]}...")  # Print only first 100 chars
                else:
                    # Single-part message might be here
                    body_data = payload.get('body', {}).get('data')
                    if body_data:
                        decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                        email_body += decoded_data
                        print(f"==> Decoded single-part email body: {decoded_data[:100]}...")  # Print only first 100 chars
                
                # Fallback to snippet if there's no body
                snippet = msg_data.get('snippet', '')
                full_content = email_body if email_body else snippet
                print(f"==> Full content of the email: {full_content[:200]}...")  # Limit log output to 200 chars

                # Classify via LLM
                labels = classify_email_with_llm(full_content)
                print(f"==> Classified labels: {labels}")
                
                labels.append(LABEL_PROCESSED)  # Add the "processed" label 
                apply_labels(service, msg['id'], labels)
                print(f"==> Applied labels to message ID: {msg['id']}")

            # If no more pages, break out of the loop
            if not next_page_token:
                break

        print(f"==> Finished fetching all messages from tab: {tab}. Total fetched: {total_fetched}")

    print("✅ Finished fetch_emails successfully!")

            

def main():
    fetch_emails()
    
if __name__ == '__main__':
    main()
