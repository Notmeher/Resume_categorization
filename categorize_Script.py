import os
import pandas as pd
import joblib
import fitz  # PyMuPDF

def load_models():
    # Load the RandomForestClassifier model
    rf_classifier = joblib.load('random_forest_model.joblib')
    # Load the TfidfVectorizer
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return rf_classifier, vectorizer

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def predict_category(text, rf_classifier, vectorizer):
    # Convert the resume summary to DataFrame
    df = pd.DataFrame([text], columns=['Resume_str'])
    
    # Transform the resume summary into TF-IDF features
    text_tfidf = vectorizer.transform(df['Resume_str'])
    
    # Make predictions
    predicted_category = rf_classifier.predict(text_tfidf)
    
    return predicted_category[0]

def categorize_pdfs(input_folder, output_csv):
    rf_classifier, vectorizer = load_models()
    
    results = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            category = predict_category(text, rf_classifier, vectorizer)
            results.append({'Filename': filename, 'Category': category})
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Categorization results saved to {output_csv}")

if __name__ == "__main__":
    input_folder = 'C:/Users/Redmibook 13/Downloads/Resume_cat/Resume_cat/dataset/Resume/Test'  # Replace with your folder path
    output_csv = 'categorized_resumes.csv'    # Replace with your desired output CSV file
    categorize_pdfs(input_folder, output_csv)
