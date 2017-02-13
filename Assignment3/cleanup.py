import pandas as pd
# Skip the first row
df = pd.read_csv ("LoanStats3a.csv", skiprows = 1)
# Get only these columns
df = df[['loan_amnt', 'term', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'loan_status']]
# Drop all records with nan
df = df.dropna()
# Only keep those with status 'Fully Paid' or 'Charged Off'
df = df[df.loan_status.isin (['Fully Paid', 'Charged Off'])]
# Discard those with emp length NA
df = df[df.emp_length != 'n/a']
# Get the first numerical values in term column (i.e. number of months)
df['term'] = df.term.apply (lambda x : int (x.split()[0]))

grades = ['G','F','E','D','C','B','A']
# Convert grade to numeric, letter grade translate to index value, numeric suffix convert to decimal values
df['gradeencoding'] = df['sub_grade'].apply (lambda x : grades.index (x[0]) + (0.7 - 0.1 * float (x[1])))
def empllengthprocess(x):
    x = x.split ('year')[0]
    if ('+') in x:
        return 12 # 10+ years = 12 years
    if ('<') in x:
        return 0 # < 1 year = 0 year
    else:
        return int (x) # Else return the value
df['emplen'] = df.emp_length.apply (lambda x : empllengthprocess (x)) # Process emplen column
# Rearrange columns
df = df[['loan_amnt', 'term', 'verification_status', 'gradeencoding', 'emplen', 'purpose', 'home_ownership', 'loan_status']]
# Print final results to CSV
df.to_csv ('Loans_processed.csv', index = False)
