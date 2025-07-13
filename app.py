import streamlit as st
import pickle
import re
# import nltk

# nltk.download("punkt")
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

nltk.download("stopwords")

# loading models
clf = pickle.load(open("clf.pkl","rb"))            # rb stands for read binary mode
tfidfd = pickle.load(open("tfidf.pkl","rb"))

def cleanResume(txt):

    cleanTxt = re.sub("http\S+\s"," ", txt)        # remove(replace with " ") urls or links from txt (jo bhi http se start hua ho or uske baad jo bhi ho space tk, agr space aya to ruk jayega)
    cleanTxt = re.sub("RT|cc"," ", cleanTxt)
    cleanTxt = re.sub("@\S+"," ", cleanTxt)        # remove emails which starts with "@" like @gmail.com
    cleanTxt = re.sub("#\S+\s"," ", cleanTxt)      # remove  "#"  or words start with "#" symbol
    cleanTxt = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_'{|}~""")," ", cleanTxt)  # removing special characters
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ", cleanTxt)
    cleanTxt = re.sub("\s+"," ", cleanTxt)

    return cleanTxt 

# Web App
def main():
    st.title("Resume Screening App")
    
    # Upload the resume file
    uploaded_file = st.file_uploader("Upload Resume", type=["txt","pdf"]) # used in uploading file (can be in txt or pdf format)

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()              # kyuki ye hmesha bytes ko hi read krta h
            resume_text = resume_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with "latin-1"
            resume_text = resume_bytes.decode("latin-1")

        cleaned_resume = cleanResume(resume_text)          # passing the resume_txt to cleanResume function to clean the resume
        input_features = tfidfd.transform([cleaned_resume])  # vectorization of resume
        prediction_id = clf.predict(input_features)[0]       # it will do the prediction
        st.write(prediction_id)

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8:  "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3:  "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6:  "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1:  "Arts",
            7:  "Database",
            11: "Electrical Engineer",
            14: "Health and fitness",
            19: "PMO",
            4:  "Business Analyst",
            9:  "DotNet Developer",
            2:  "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5:  "Civil Engineer",
            0:  "Advocate",

        }
        category_name = category_mapping.get(prediction_id, "Unknown") # iska mtlb prediction_id k hisab se name return krna h or agr unme se koi nhi hua to "Unknown" return krna h

        st.write("Predicted Category:", category_name)

# Python main
if __name__ == "__main__":
    main()