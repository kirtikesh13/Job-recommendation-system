# JOB RECOMMENDATION SYSTEM - DASHBOARD UI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time

# Page config
st.set_page_config(page_title="Job Recommendation System", page_icon="💼", layout="wide")

# CSS Styling
st.markdown("""
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background-color: #f4f9ff;
    background-image: linear-gradient(to right, red, orange);
}

/* Title */
h1 {
    text-align: center;
    color: #1e3a8a;
    font-family: 'Poppins', sans-serif;
    padding-bottom: 0.2em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e3a8a;
}
[data-testid="stSidebar"] h2 {
    color: white;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stSidebar"] span, label, p {
    color: #e0e7ff;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
div.stButton > button:hover {
    background-color: red;
}

/* Job cards */
.job-card {
    background-color: white;
    padding: 1rem;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.job-title {
    font-weight: bold;
    color: #1e3a8a;
    font-size: 18px;
}
.job-company {
    color: #2563eb;
    font-size: 16px;
}
.job-location {
    color: #475569;
    font-size: 14px;
}
.match-score {
    color: green;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("final_data.csv")
    return df

df = load_data()

skill_columns = ['PYTHON','C++','JAVA','HADOOP','SCALA','FLASK','PANDAS','SPARK',
                 'NUMPY','PHP','SQL','MYSQL','CSS','MONGODB','NLTK','TENSORFLOW',
                 'LINUX','RUBY','JAVASCRIPT','DJANGO','REACT','REACTJS','AI','UI',
                 'TABLEAU','NODEJS','EXCEL','POWER BI','SELENIUM','HTML','ML']

info_columns = ['Company_Name','Designation','Location','Industry','Level','Involvement']

skill_matrix = df[skill_columns].values

# Recommendation Function
def recommend_jobs(user_skills, top_n=5):
    user_vector = [1 if skill.upper() in [s.upper() for s in user_skills] else 0 for skill in skill_columns]
    similarities = cosine_similarity([user_vector], skill_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices][info_columns].copy()
    recommendations["Match_Score"] = similarities[top_indices]
    return recommendations

# Sidebar
st.sidebar.title("💼 Job Recommender")
st.sidebar.write("Find your best-matching jobs based on your skills.")

# Input Section
st.sidebar.subheader("Enter Your Skills")
user_input = st.sidebar.text_area("e.g., Python, SQL, ML, Excel")

top_n = st.sidebar.slider("Number of job results", 5, 10, 20, 5)

# Main Page
st.title("🚀 Smart Job Recommendation Dashboard")
st.markdown("### Discover your ideal job match based on your unique skill set!")

if st.sidebar.button("🔍 Recommend Jobs"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter at least one skill.")
    else:
        user_skills = [s.strip() for s in user_input.split(",")]
        with st.spinner("Matching your skills with job listings..."):
            time.sleep(1)
            results = recommend_jobs(user_skills, top_n)
            st.success(f"✅ Top {top_n} Job Matches for: {', '.join(user_skills)}")

            # Job Cards
            for _, row in results.iterrows():
                st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{row['Designation']}</div>
                        <div class="job-company">{row['Company_Name']}</div>
                        <div class="job-location">📍 {row['Location']} | 🏭 {row['Industry']}</div>
                        <div class="job-location">👔 Level: {row['Level']} | ⏰ {row['Involvement']}</div>
                        <div class="match-score">⭐ Match Score: {round(row['Match_Score']*100, 2)}%</div>
                    </div>
                """, unsafe_allow_html=True)

            # Download CSV
            csv = results.to_csv(index=False)
            st.download_button("⬇️ Download Results as CSV", csv, "recommended_jobs.csv", "text/csv")

# Footer
st.markdown("---")
st.caption("Built by Kirtikesh | Powered by Streamlit & Data Science")
