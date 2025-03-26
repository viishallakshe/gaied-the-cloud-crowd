import os
import email
import pandas as pd
from email import policy
from email.parser import BytesParser
from PIL import Image
import pytesseract
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import pipeline

# Function to extract email content from .eml file
def extract_email_content(eml_file):
    with open(eml_file, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    
    # Extracting plain text content
    text_content = ""
    for part in msg.iter_parts():
        if part.get_content_type() == "text/plain":
            text_content = part.get_payload(decode=True).decode(part.get_content_charset())
    
    return text_content

# Function to handle OCR on image attachments (using Tesseract)
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Preprocessing the CSV data for training
def preprocess_csv(csv_file):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Map string labels to numeric values
    label_map = {"AUTransfer": 0, "FeePayment": 1, "ClosingNotice": 2}
    df['label'] = df['label'].map(label_map)

    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert DataFrame into Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

# Tokenize the dataset
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True)

# Train the classification model
def train_model(train_dataset, test_dataset):
    # Initialize the model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

    # Tokenize the datasets
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    return model, tokenizer

# Use the trained model to classify new emails
def classify_email(text, model, tokenizer):
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(text)
    return result

# Function to process multiple .eml files from a directory and classify them
def classify_multiple_emails(eml_directory, model, tokenizer):
    # List all .eml files in the directory
    eml_files = [f for f in os.listdir(eml_directory) if f.endswith(".eml")]
    
    # Iterate over each .eml file
    for eml_file in eml_files:
        eml_path = os.path.join(eml_directory, eml_file)
        print(f"\nClassifying email: {eml_file}")
        
        # Extract email content
        email_content = extract_email_content(eml_path)
        
        # If there are attachments, extract text using OCR (for image attachments)
        # You can modify this part as needed to extract attachments dynamically
        attachment_path = os.path.join(eml_directory, f"{eml_file}.png")  # Example attachment path
        if os.path.exists(attachment_path):
            ocr_text = extract_text_from_image(attachment_path)
            email_content += "\n" + ocr_text  # Append OCR text to email content
        
        # Classify the email
        prediction = classify_email(email_content, model, tokenizer)
        
        # Output the classification result
        print(f"Prediction for {eml_file}: {prediction}")

# Main function to preprocess CSV and train the model, then classify emails
def main():
    # Preprocess CSV file and train the model
    train_dataset, test_dataset = preprocess_csv('E:/Hackathon 2025/Hackathon 2025 Cloud Crowd Final/Emails.csv')  # Use your actual CSV file path
    model, tokenizer = train_model(train_dataset, test_dataset)

    # Directory containing the .eml files
    eml_directory = 'E:/Hackathon 2025/Hackathon 2025 Cloud Crowd Final/Sample Email Files'  # Update this with your actual directory containing .eml files

    # Classify all emails in the directory
    classify_multiple_emails(eml_directory, model, tokenizer)

if __name__ == "__main__":
    main()
