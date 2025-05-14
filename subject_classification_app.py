
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from supabase import create_client
import torch

# Configuration and Initialization 
def get_config():
    if st.secrets:  # Production (Streamlit Community Cloud)
        return {
            "supabase_url": st.secrets["SUPABASE_URL"],
            "supabase_key": st.secrets["SUPABASE_KEY"],
            "api_key": st.secrets["API_KEY"]
        }
    else:  # Local development
        from dotenv import load_dotenv
        load_dotenv()
        return {
            "supabase_url": os.getenv("SUPABASE_URL"),
            "supabase_key": os.getenv("SUPABASE_KEY"),
            "api_key": os.getenv("API_KEY")
        }

# Initialize services
config = get_config()
supabase = create_client(config["supabase_url"], config["supabase_key"])
genai.configure(api_key=config["api_key"])


def format_text(input_text):
    try:
        prompt = f"""Convert this unstructured text into a clean, comma-separated list of university subjects:

        Original Text:
        {input_text}

        Rules:
        1. Extract only subject names
        2. Remove duplicates
        3. Trim extra spaces
        5. Maintain original subject casing
        6. Capitalize the First Letter of Each Word 
        7. Output ONLY the comma-separated list"""

        # Create model instance with specified model
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.1
            )
        )

        raw_output = response.text.strip()
        return ', '.join([s.strip() for s in raw_output.split(',')])

    except Exception as e:
        st.error(f"Formatting failed: {str(e)}")
        return ""

SUBJECT_MAPPING = {
        # Business Statistics
    "Business Statistics I": "Business Statistics",
    "Business Statistics II": "Business Statistics",
    "Statistical Data Analysis": "Business Statistics",
    "Statistics for Analytics I": "Business Statistics",
    "Statistics for Analytics II": "Business Statistics",
    "Probability and Statistics for Business - I": "Business Statistics",
    "Probability and Statistics for Business - II": "Business Statistics",
    "Statistical Simulation": "Business Statistics",
    "Applied Statistical Computing": "Business Statistics",
    "Statistical Inference": "Business Statistics",

    # Operations Management
    "Operation Management": "Operations Management",
    "Operations Management": "Operations Management",
    "Advanced Operations Research": "Operations Research",
    "Operations Research I": "Operations Research",
    "Operations Research II": "Operations Research",
    "Operations Research": "Operations Research",
    "Introduction to Operations Research": "Operations Research",

    # Data Mining
    "Data Mining and Warehousing": "Data Mining",
    "Data Mining and Predictive Analytics": "Data Mining",
    "Data Mining": "Data Mining",

    # Business Communication
    "Business Communication Skills I": "Business Communication",
    "Business Communication Skills II": "Business Communication",
    "Fundamentals of Business Communication": "Business Communication",
    "Communication Skills": "Business Communication",
    "Analytical Writing Skills": "Business Communication",
    "Technical & Scientific Writing": "Business Communication",
    "Technical & Academic Writing": "Business Communication",
    "Advanced Communication Skills": "Business Communication",
    "Conversation Analysis": "Business Communication",
    "Introduction to Communication Skills": "Business Communication",
    "Business Discourse Analysis": "Business Communication",

    # Data Visualization
    "Data Visualization using Tableau": "Data Visualization",
    "Data Visualization with Power BI": "Data Visualization",
    "Data Visualization": "Data Visualization",
    "Data Management & Visualization": "Data Visualization",

    # Financial Accounting
    "Introduction to Accounting": "Financial Accounting",
    "Financial Accounting": "Financial Accounting",
    "Accounting and Finance": "Financial Accounting",
    "Cost & Management Accounting": "Financial Accounting",
    "Cost and Management Accounting": "Financial Accounting",
    "Auditing and Taxation": "Financial Accounting",

    # Principles of Management
    "Management Process": "Principles of Management",
    "Organization Behaviour & Management": "Principles of Management",
    "Principles of Management": "Principles of Management",

    # Microeconomics and Macroeconomics
    "Micro-economics": "Microeconomics",
    "Microeconomics": "Microeconomics",
    "Macroeconomics": "Macroeconomics",

    # Human Resource Management
    "Human Resource Analytics": "Human Resource Management",
    "Human Resource Management": "Human Resource Management",

    # Programming
    "Programming Fundamentals": "Fundamentals of Programming",
    "Principles of Programming": "Fundamentals of Programming",
    "Programming Laboratory": "Fundamentals of Programming",
    "Programming for Data Science": "Fundamentals of Programming",

    # Database Management
    "Database Management Systems": "Database Management",
    "Advanced Database Management": "Database Management",
    "Advanced Database Management Systems": "Database Management",
    "Fundamentals of Databases": "Database Management",
    "Database Management": "Database Management",

    # Project Management
    "Project Management for Data Science": "Project Management",
    "Project Management": "Project Management",

    # Research Methods
    "Research Methodology": "Research Methods",
    "Research Methods": "Research Methods",
    "Research Writing Skills": "Research Methods",
    "Comprehensive Research Project(Continued to Semester 2)": "Research Project",
    "Research Project": "Research Project",

    # Business Law
    "Legal & Political Environment in Business": "Business Law",
    "Business Law": "Business Law",

    # Strategic Management
    "Global Strategy": "Strategic Management",
    "Strategic Analytics": "Strategic Management",
    "Strategic Management": "Strategic Management",

    # ERP
    "Enterprise Resource Planning (ERP)": "Enterprise Resource Planning",
    "Enterprise Resource Planning": "Enterprise Resource Planning",

    # Marketing
    "Principles of Marketing": "Marketing Management",
    "Marketing Analytics": "Marketing Management",
    "Marketing Management": "Marketing Management",

    # Business Analytics
    "Foundations of Business Analytics": "Business Analytics",
    "Introduction to Business Analytics": "Business Analytics",
    "Business Metrics for Data Driven Companies": "Business Analytics",
    "Strategic Business Analysis": "Business Analytics",
    "Business Analytics Tools and Technologies": "Business Analytics",
}

try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    st.warning(f"GPU not available, falling back to CPU: {str(e)}")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')


# Precompute canonical embeddings
canonical_subjects = list(set(SUBJECT_MAPPING.values()))
canonical_embeddings = embedding_model.encode(canonical_subjects, convert_to_tensor=True)

def get_canonical_subject(subject_name):
    if not subject_name or not subject_name.strip():
        return "Unknown"
    
    subject_name_cleaned = subject_name.lower().strip()
    
    # First check substring matches
    for variant, canonical in SUBJECT_MAPPING.items():
        if variant.lower() in subject_name_cleaned:
            return canonical
    
    # Compute embedding similarity
    subject_embedding = embedding_model.encode(subject_name_cleaned, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(subject_embedding, canonical_embeddings)[0]
    
    best_score = float(cosine_scores.max())
    best_idx = int(cosine_scores.argmax())

    if best_score > 0.75:
        return canonical_subjects[best_idx]
    
    # Normalize generic terms
    remove_phrases = ["fundamentals of", "introduction to", "advanced", "basic", "Ii","I"]
    for phrase in remove_phrases:
        subject_name_cleaned = subject_name_cleaned.replace(phrase, "").strip()
    
    return subject_name_cleaned.title() or "Unknown"


def classify_subject(subject_name):
    categories = [
        "Business & Management",
        "Quantitative & Analytical Foundations",
        "Business Analytics & Data Science",
        "Information Systems & Technology",
        "Digital & Innovation",
        "Communication & Professional Development",
        "Research",
        "Experience"
    ]

    prompt = f"""
    You are a genius in the business analytics field and have deep understanding of the theoretical subjects that are essential for the industry,
    including those that support descriptive, predictive, and optimization analytics. 
    Based on your expertise classify the university subject "{subject_name}" into one of these categories: 
    {', '.join(categories)}.
    Return ONLY the category name. If it doesn't match any category, return "Uncategorized".
     Examples:
    Learning and Study Skills → Communication & Professional Development
    Principles of Management → Business & Management
    Microeconomics → Business & Management
    Business Mathematics → Quantitative & Analytical Foundations
    Information Technology for Business → Information Systems & Technology
    English Language Skills → Communication & Professional Development
    Self Management → Communication & Professional Development
    Macroeconomics → Business & Management
    Financial Accounting → Business & Management
    Legal & Political Environment in Business → Business & Management
    Human Resources Management → Business & Management
    Business Communication → Communication & Professional Development
    Personal Development Planning → Communication & Professional Development
    Organizational Behavior → Business & Management
    Business Information Systems → Information Systems & Technology
    Principles of Marketing → Business & Management
    Business Statistics → Quantitative & Analytical Foundations
    Operations Management → Business & Management
    Leadership and Teamwork → Business & Management
    Business Negotiation → Business & Management
    Foundations of Business Analytics → Business Analytics & Data Science
    Mastering Data Analysis in Excel → Business Analytics & Data Science
    Database Management Systems → Business Analytics & Data Science
    Digital Strategy and Innovation → Digital & Innovation
    Customer Analytics → Business Analytics & Data Science
    Data Science in Real Life → Business Analytics & Data Science
    Data Visualisation → Business Analytics & Data Science
    Career Readiness and Business Etiquettes → Communication & Professional Development
    Project Management → Business & Management
    Business Ethics and Values → Business & Management
    R and Python Programming → Business Analytics & Data Science
    Operations Analytics → Business Analytics & Data Science
    Business Research Methods → Research
    Business Internship → Experience
    Comprehensive Research Project(Continued to Semester 2) → Research
    Strategic Management → Business & Management
    Business Metrics for Data Driven Companies → Business Analytics & Data Science
    Social Media and Web Analytics → Digital & Innovation
    Information Systems Management and Security → Information Systems & Technology
    Supply Chain Analytics → Business Analytics & Data Science
    Decision Modeling for Business Analytics → Business Analytics & Data Science
    Data Mining and Predictive analytics → Business Analytics & Data Science
    Accounting Analytics → Business Analytics & Data Science
    People Analytics → Business Analytics & Data Science
    Enterprise Resource → Enterprise Resource Planning (ERP)

    """

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate response
        response = model.generate_content(
            prompt,
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=500,
                temperature=0.5
            )
        )

        predicted_category = response.text.strip()
        return predicted_category if predicted_category in categories else "Uncategorized"

    except Exception as e:
        st.error(f"Classification failed for '{subject_name}': {str(e)}")
        return "Uncategorized"

def check_subject_in_kiu(subject_name):
    KIU_SUBJECTS = [
    "Business Statistics I", "Business Mathematics I", "Programming Fundamentals", "Introduction to Accounting",
    "Microeconomics", "Business Communication Skills I", "Business Mathematics II", "Business Statistics II",
    "Macroeconomics", "Principles of Management", "Data Structures & Algorithms", "Business Communication Skills II",
    "Business Mathematics III", "Econometrics", "Financial Management", "Database Management", "Organizational Behaviour",
    "Operations Research I", "Corporate Finance", "Operations Management", "Data Visualization", "System Analysis & Design",
    "Technical & Academic Writing", "Operations Research II", "Machine Learning - I", "Principles of Marketing",
    "Enterprise Resource Planning", "Business Process Management", "Machine Learning II", "Supply Chain Management",
    "Project Management", "Human Resource Management", "Financial Modelling", "Business Application Development",
    "Research Methods", "Advanced Database Management", "Investment & Portfolio Management", "Strategic Management",
    "Business Internship", "Big Data Technologies", "Business Process Automations", "Business Law",
    "Data Privacy and Ethics", "Research Project"]

    try:
        kiu_subjects = ", ".join([s.strip() for s in KIU_SUBJECTS])

        prompt = f"""
        Check if the subject "{subject_name}" already exists in this list of subjects taught at KIU:
        [{kiu_subjects}]

        Instructions:
        - Be tolerant of small differences in phrasing, punctuation, or synonyms.
        - If the details of '{subject_name}' are covered or interconnected in [{kiu_subjects}], just write 'Yes'. As some subjects offer the same content under different names.
        - Consider a subject equivalent if it's conceptually the same (e.g., "Intro to Accounting" ≈ "Introduction to Accounting").
        - Respond with ONLY "Yes" or "No".
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=5,
                temperature=0.2
            )
        )

        result = response.text.strip().lower()
        return "Yes" if "yes" in result.lower().strip() else "No"

    except Exception as e:
        st.error(f"KIU Check failed for '{subject_name}': {str(e)}")
        return "No"

def save_to_supabase(data):
    supabase = get_config()
    supabase = create_client(config["supabase_url"], config["supabase_key"])
    
    try:
        # Prepare data for insertion
        records = []
        for item in data:
            records.append({
                "country": item["Country"],
                "university": item["University"],
                "year": item["Year"],
                "semester": item["Semester"],
                "course_type": item["Core or Elective"],
                "subject": item["Subject"],
                "category": item["Category"],
                "offered_at_kiu": item["Offered at KIU"],
                "canonical_subject": item["Canonical Subject"]
            })
        
        # Insert records
        response = supabase.table("subjects").insert(records).execute()
        
        if hasattr(response, 'error') and response.error:
            raise Exception(response.error)
            
        return f"Successfully saved {len(records)} records to Supabase"
    
    except Exception as e:
        raise Exception(f"Supabase save failed: {str(e)}")

def display_unique_counts():
    supabase = get_config()
    supabase = create_client(config["supabase_url"], config["supabase_key"])
    
    try:
        # Fetch unique categories
        categories_res = supabase.rpc('get_unique_categories').execute()
        unique_categories = [item['category_name'] for item in categories_res.data] if categories_res.data else []
        
        # Fetch unique subjects
        subjects_res = supabase.rpc('get_unique_subjects').execute()
        unique_subjects = [item['subject_name'] for item in subjects_res.data] if subjects_res.data else []
        
        # Display in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Categories")
            st.write(f"**Total Unique Categories:** {len(unique_categories)}")
            st.write("**All Categories:**")
            st.write(pd.DataFrame(sorted(unique_categories), columns=["Category"]))
        
        with col2:
            st.subheader("Canonical Subjects")
            st.write(f"**Total Unique Subjects:** {len(unique_subjects)}")
            st.write("**All Subjects:**")
            st.write(pd.DataFrame(sorted(unique_subjects), columns=["Subject"]))
            
    except Exception as e:
        st.error(f"Error loading data from Supabase: {str(e)}")
            
def main():

    if st.session_state.get("trigger_refresh"):
        st.session_state.clear()
        st.rerun()
    
    # Step 2: Initialize all fields if not already
    defaults = {
        "formatted_subjects": "",
        "classified_data": [],
        "edited_data": [],
        "country": "",
        "university": "",
        "year": "Year 1",
        "semester": "Sem 1",
        "course_type": "Core",
        "raw_text": ""
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # UI Setup
    st.markdown("<div style='text-align: center;'><h2>Business Analytics Subject Classification App</h1></div>", unsafe_allow_html=True)
    st.image("images/background.png")

    if st.button("🔄 Refresh App"):
        st.session_state["trigger_refresh"] = True
        st.rerun()

    with st.form("subject_form"):
        st.markdown("<h4>Enter Subject Details</h3>", unsafe_allow_html=True)
        
        # Text formatting section
        with st.expander("📝 Paste Unformatted Text"):
            st.session_state.raw_text = st.text_area("Input unformatted subject list", height=150, value=st.session_state.raw_text)
            if st.form_submit_button("✨ Clean and Format Subjects"):
                if st.session_state.raw_text:
                    with st.spinner("Formatting..."):
                        formatted = format_text(st.session_state.raw_text)
                        st.session_state.formatted_subjects = formatted
                else:
                    st.warning("Please input text to format")
        
        # Input fields
        st.session_state.country = st.text_input("Country", value=st.session_state.country)
        st.session_state.university = st.text_input("University", value=st.session_state.university)
        st.session_state.year = st.selectbox("Year", ["Year 1", "Year 2", "Year 3", "Year 4","Unknown"], index=["Year 1", "Year 2", "Year 3", "Year 4","Unknown"].index(st.session_state.year))
        st.session_state.semester = st.selectbox("Semester", ["Sem 1", "Sem 2","Sem 3", "Sem 4", "Sem 5", "Sem 6","Sem 7", "Sem 8","Unknown"], index=["Sem 1", "Sem 2", "Sem 3", "Sem 4", "Sem 5", "Sem 6", "Sem 7", "Sem 8","Unknown"].index(st.session_state.semester))
        st.session_state.course_type = st.selectbox("Core or Elective", ["Core", "Elective"], index=["Core", "Elective"].index(st.session_state.course_type))

        subjects = st.text_area("Subjects (comma-separated)", value=st.session_state.formatted_subjects, height=100)
        
        col1, col2 = st.columns(2)
        with col1:
            classify_btn = st.form_submit_button("🏷️Classify Subjects")
        with col2:
            save_btn = st.form_submit_button("💾 Save to Database")

        if classify_btn:
            if not all([st.session_state.country, st.session_state.university, subjects]):
                st.error("Please fill all required fields")
            else:
                subject_list = [s.strip() for s in subjects.split(',')]
                st.session_state.classified_data = []
                
                with st.spinner("Classifying subjects..."):
                    for subject in subject_list:
                        category = classify_subject(subject)
                        canonical = get_canonical_subject(subject)
                        kiu_status = check_subject_in_kiu(subject)
                        
                        st.session_state.classified_data.append({
                            "Subject": subject,
                            "AI Suggestion": category,
                            "Final Category": category.split("→ Suggest:")[-1].strip() if "→ Suggest:" in category else category,
                            "Offered at KIU": kiu_status,
                            "Canonical Subject": canonical,
                            "Count": 0  # Will be updated when saved
                        })
                
                st.session_state.edited_data = st.session_state.classified_data.copy()

        # Display editable table
        if st.session_state.edited_data:
            st.subheader("Review Classifications")
            
            edited_df = pd.DataFrame(st.session_state.edited_data)
            edited_df = st.data_editor(
                edited_df,
                column_config={
                    "AI Suggestion": st.column_config.TextColumn("AI Suggestion", disabled=True),
                    "Final Category": st.column_config.TextColumn("Final Category", help="Edit the category as needed"),
                    "Offered at KIU": st.column_config.TextColumn("Offered at KIU"),
                    "Count": None,
                    "Canonical Subject": st.column_config.TextColumn("Canonical Subject")  # Hide from view
                },
                hide_index=True,
                use_container_width=True,
                key="category_editor"
            )
            
            st.session_state.edited_data = edited_df.to_dict('records')

        if save_btn and st.session_state.edited_data:
            try:
                final_data = []
                for item in st.session_state.edited_data:
                    final_data.append({
                        "Country": st.session_state.country,
                        "University": st.session_state.university,
                        "Year": st.session_state.year,
                        "Semester": st.session_state.semester,
                        "Core or Elective": st.session_state.course_type,
                        "Subject": item["Subject"],
                        "Category": item["Final Category"],
                        "Offered at KIU": item.get("Offered at KIU", "No"),
                        "Canonical Subject": item.get("Canonical Subject", get_canonical_subject(item["Subject"])),
                    })
                
                result = save_to_supabase(final_data)
                st.success(result)
                
                # Reset states
                st.session_state.classified_data = []
                st.session_state.edited_data = []
                st.session_state.formatted_subjects = ""
                st.session_state.country = ""
                st.session_state.university = ""
                st.session_state.year = "Year 1"
                st.session_state.semester = "Sem 1"
                st.session_state.course_type = "Core"
                st.session_state.raw_text = ""

                st.success(result)
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Save failed: {str(e)}")

        # Update the display section
        with st.expander("📊 View Existing Subject Categories", expanded=False):
            display_unique_counts()

if __name__ == "__main__":
    main()