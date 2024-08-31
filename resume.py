import pandas as pd
import joblib

def load_models():
    # Load the RandomForestClassifier model
    rf_classifier = joblib.load('random_forest_model.joblib')
    # Load the TfidfVectorizer
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    return rf_classifier, vectorizer

def predict_category(resume_summary):
    # Load the models
    rf_classifier, vectorizer = load_models()

    # Convert the resume summary to DataFrame
    new_resume_df = pd.DataFrame([resume_summary], columns=['Resume_str'])

    # Transform the resume summary into TF-IDF features
    new_resume_tfidf = vectorizer.transform(new_resume_df['Resume_str'])

    # Make predictions
    predicted_category = rf_classifier.predict(new_resume_tfidf)

    return predicted_category[0]

if __name__ == "__main__":
    # Example resume summary
    resume_summary = """Results driven and award winning accounting and auditing professional with over ten years of experience. 
Motivated team leader and excellent
mentor. 
Exceptional skills in: Generally Accepted Accounting Principles 
Generally Accepted Auditing Standards Interviewing Techniques 
Research
& Data Gathering Financial Analysis 
Budget Preparation Accounts Payable/Accounts Receivable 
Implementing Effective Internal Controls OMB
Circular A-133 
MS Office
Experience
09/2015
to 
03/2016
Accountant
Company Name
ï¼​ 
City, 
State
Evaluated the effectiveness of financial processes, and made procedural changes to improve Child Support Cashier Collections and
Settlement which, decreased incidence of misallocated funds due to software glitches and documentation mistakes.
Recommended video surveillance system installation which, safeguarded cashiers.
Participated in committees and task forces established to analyze and resolve systemic problems.
Performed a financial data analysis of $1.3 million and 4,700 receipts to locate busy periods in the Child Support Cashier Booth, and noted
that the Cashier Booth is usually the most busy during the beginning and end of the month as well as around holidays.
Interpreted and implemented a variety of policies and guidelines, and proposed findings and solutions to decrease errors and susceptibility to
fraud such as utilizing a second cash register at the Child Support Cashier Booth when cashiers may feel overwhelmed from high client
volume.
Supervised and trained six staff members on proper utilization of policies and procedures to insure that there would be less possibility of
inaccuracies.
Reviewed and approved twelve bank reconciliations to insure accuracy.
Reconciled Child Support Fee check register on a daily and monthly basis.
Audited Child Support Files, and reconciled amount of Child Support owed to the PACSES (Pennsylvania Child Support Enforcement
System),.
09/2014
to 
09/2015
Administrative Specialist II
Company Name
ï¼​ 
City, 
State
Posted and entered accounting data to the City's financial database using proper fund codes.
Gathered and reviewed supporting detail as well as re-computed invoices and backup documentation, and authorized invoices for payment.
Examined accounting records to ensure that all data was correctly and consistently recorded.
Identified and corrected incorrect entries and other clerical errors.
Also, communicated with vendors to assist with billing disputes.
Prepared bank deposits and booked income for SELF Inc.
client savings accounts in Quickbooks.
Also, recorded receipt of money orders.
Reviewed and evaluated target, operating, and quarterly budgets for funds amounting to approximately $100 M.
Developed reports required by Federal and State monitoring agencies for the Homeless Prevention and Rapid Re-Housing (HPRP) and
Child and Adult Care Food (CACFP) program within deadlines.
Reviewed relevant regulations, contracts, laws, ordinances and procedures governing departmental decision-making.
04/2005
to 
09/2014
Auditor II
Company Name
ï¼​ 
City, 
State
Evaluated city departments for conformity with SAPS (Standard Accounting Procedures), GAAP (Generally Accepted Accounting
Principles), and departmental policies and procedures.
Audited city departments for grant compliance with CFDA (Catalog of Financial Domestic Assistance) and state requirements.
Assessed Federal and State grant compliance requirements including: 
laws and regulations, administrative procedures, contract terms, and
general grant stipulations.
Reviewed prior year's audit documentation, audit report, management letter, and budgetary testimony.
Met with department representatives during entrance conferences to discuss objectives and timetables.
Interviewed department officials to gain knowledge of the internal control systems in place.
Also, prepared internal control questionnaires to assist in the modification of audit programs.
Selected audit sample, and completed attribute testing of sample items.
Prepared and organized audit work papers.
Formulated findings and recommendations based on exceptions found.
Investigated cases of suspected fraud or abuse including: 
noncompliance with Charter School laws, contractual fraud, co-mingling of funds,
and overstatement of assets on financial statements.
01/2003
to 
04/2005
Accountant
Company Name
ï¼​ 
City, 
State
Searched account histories to locate imbalances and incorrect entries.
Prepared invoices and reconciled asset and liability account balances for 300 agency contracts, which accounted for $550 M per fiscal year"""

    predicted_category = predict_category(resume_summary)
    print("Predicted Category:", predicted_category)
