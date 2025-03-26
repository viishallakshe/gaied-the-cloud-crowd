import os
import email
from email import policy
from email.parser import BytesParser
from transformers import pipeline
import pytesseract
import pytesseract
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize Hugging Face zero-shot classification pipeline
classifier = pipeline("zero-shot-classification")

# Define possible labels for classification
RequestType_labels = ["Closing Notice", "Commitment Change", "Fee Payment", "Money Movement-Inbound", "Money Movemnt-Outbound"]

# Function to parse .eml file and extract content
def parse_eml(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Extract the subject, sender, and body (plain text)
    subject = msg['subject']
    sender = msg['from']
    body = msg.get_body(preferencelist=('plain')).get_payload(decode=True).decode()

    return subject, sender, body, msg

# Function to classify email using Hugging Face Zero-Shot Classification
def classify_email(body):
    result = classifier(body, RequestType_labels)
    label = result['labels'][0]  # Get the label with the highest score
    return label

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image_data):
    image = Image.open(BytesIO(image_data))  # Convert binary data to an image
    text = pytesseract.image_to_string(image)  # Use Tesseract to extract text
    return text

# Function to process email attachments (OCR)
def process_attachments(msg):
    extracted_text = ""
    for part in msg.iter_attachments():
        # If the attachment is an image, try to extract text
        if part.get_content_type().startswith('image'):
            print(f"Processing image attachment: {part.get_filename()}")
            image_data = part.get_payload(decode=True)  # Get image data in binary format
            extracted_text += extract_text_from_image(image_data) + "\n"  # Extract text and add to result
    return extracted_text

# Main function to process and classify multiple .eml files
def process_email(file_path):
    # Parse the email
    subject, sender, body, msg = parse_eml(file_path)
    print(f"\nProcessing Email: {file_path}")
    print(f"Email Subject: {subject}")
    print(f"Email From: {sender}")
    
    # Classify the email content
    classification = classify_email(body)
    print(f"Email classified as: {classification}")
    
    # Process any attachments for OCR
    extracted_text = process_attachments(msg)
    if extracted_text:
        print(f"\nText extracted from image attachments:\n{extracted_text}")
    else:
        print("No image attachments found.")

# Function to process all emails in a directory
def process_multiple_emails(directory_path):
    if not os.path.isdir(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith('.eml'):
            file_path = os.path.join(directory_path, filename)
            process_email(file_path)
        else:
            print(f"Skipping non-.eml file: {filename}")

# Example usage
directory_path = 'E:/Hackathon 2025/Hackathon 2025 Cloud Crowd Final/Sample Email Files'  # Replace with your directory containing .eml files
process_multiple_emails(directory_path)
