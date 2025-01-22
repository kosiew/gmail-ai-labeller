import os
import pickle
import subprocess
import base64

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

GPT4ALL_MODEL="orca-mini-3b-gguf2-q4_0"
LABELS=["programming", "news", "machine_learning", "etc"]

def authenticate_gmail():
    """
    Authenticate to Gmail using OAuth. 
    Stores/loads credentials in 'token.pickle'.
    """
    SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
    creds = None
    
    # Load stored credentials if available
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate again
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
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
        prompt = (
            "Here are some labels.\n"
            f"{labels}\n"
            "Please return a list of applicable labels.\n"
            f"for this content:{content}"
        )
        
        result = subprocess.run(
            ["llm", "complete", "--model", GPT4ALL_MODEL, prompt],
            capture_output=True,
            text=True
        )

        classification = result.stdout.strip()
        # You could parse the classification more precisely if needed.
        # For simplicity, we'll just check if "Archive" is in the output.
        # Adjust logic depending on how the model responds.
        
        # classification is labels delimited with comma, 
        # return a list of labels
        classification = classification.split(",")
        return classification

    except Exception as e:
        # If an error occurs, default to "Do Not Archive"
        return ["error"]

def apply_labels(service, msg_id, labels):
    """
    Create (if needed) and apply Gmail labels to the given message.
    """
    # Get existing labels
    gmail_labels = service.users().labels().list(userId='me').execute().get('labels', [])
    
    label_ids = []
    
    for label_name in labels:
        label_id = None
        
        # Check if label exists
        for label in gmail_labels:
            if label['name'] == label_name:
                label_id = label['id']
                break
        
        # If not, create a new label
        if not label_id:
            new_label = service.users().labels().create(
                userId='me',
                body={
                    "name": label_name,
                    "labelListVisibility": "labelShow",
                    "messageListVisibility": "show"
                }
            ).execute()
            label_id = new_label['id']
        
        label_ids.append(label_id)
    
    # Apply all labels to the email at once
    service.users().messages().modify(
        userId='me',
        id=msg_id,
        body={'addLabelIds': label_ids}
    ).execute()

def fetch_emails(max_results=10):
    """
    Fetch emails from Gmail inbox, classify them, and label if necessary.
    Returns a dictionary of { "Archive": [...], "Do Not Archive": [...] } 
    containing snippets or partial contents for convenience.
    """
    # Authenticate and build service
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)
    
    TABS = ["INBOX", "CATEGORY_SOCIAL", "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"]

    for tab in TABS:
        results = service.users().messages().list(
            userId='me', 
            labelIds=[tab], 
            maxResults=max_results
        ).execute()
        messages = results.get('messages', [])
        
        for msg in messages:
            msg_data = service.users().messages().get(
                userId='me', 
                id=msg['id'], 
                format='full'   # 'full' to get the entire message payload
            ).execute()
            
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
            else:
                # Single-part message might be here
                body_data = payload.get('body', {}).get('data')
                if body_data:
                    decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                    email_body += decoded_data
            
            # Fallback to snippet if there's no body
            snippet = msg_data.get('snippet', '')
            full_content = email_body if email_body else snippet

            # Classify via LLM
            labels = classify_email_with_llm(full_content)
        
            apply_labels(service, msg['id'], labels)
            

def main():
    categorized_emails = fetch_emails(max_results=10)
    
    print("\nCategorized Emails:")
    for category, emails in categorized_emails.items():
        print(f"\n{category} ({len(emails)}):")
        for email_text in emails:
            print(f"  - {email_text[:100]}...")  # Print a portion for demonstration

if __name__ == '__main__':
    main()
