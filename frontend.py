from google import genai
import streamlit as st
import os, json, hashlib, re
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import pandas as pd

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# ---------------- CONFIG ----------------
USER_DB = "users.json"
JD_FOLDER = "job_descriptions"
os.makedirs(JD_FOLDER, exist_ok=True)

COMMON_SKILLS = [
    "Python","Java","C++","C#","JavaScript","TypeScript","HTML","CSS",
    "SQL","NoSQL","PostgreSQL","MongoDB","MySQL","Oracle",
    "Docker","Kubernetes","AWS","Azure","Google Cloud","GCP","Cloud",
    "Git","GitHub","GitLab","Jira","CI/CD",
    "React","Angular","Vue.js","Node.js","Express.js",
    "Django","Flask","Spring","ASP.NET",
    "Data Analysis","Machine Learning","Deep Learning","NLP",
    "TensorFlow","PyTorch","Scikit-learn","Pandas","NumPy","Matplotlib",
    "Power BI","Tableau","Excel","Data Visualization",
    "Agile","Scrum","Kanban","Project Management",
    "REST API","Microservices","System Design",
    "Cybersecurity","Network Security","Penetration Testing"
]

GENERIC_JDS = {
    "Data Analyst": "Experienced data analyst with strong skills in Python, SQL, and Power BI/Tableau for data visualization.",
    "Full Stack Developer": "Proficient in JavaScript, HTML, CSS, React, Node.js, Express.js, Git, and cloud platforms.",
    "Cloud Architect": "Expert in cloud solutions (AWS/Azure/GCP), Docker, Kubernetes, CI/CD pipelines.",
    "Machine Learning Engineer": "Strong Python, ML algorithms, TensorFlow/PyTorch, NLP experience, deployment skills.",
    "Cybersecurity Analyst": "Experience in network security, penetration testing, incident response, and cybersecurity best practices.",
    "General Purpose": "Experienced professional with strong Python, SQL, Cloud, Data Analysis, ML, NLP, Git, Agile, project management, and visualization skills."
}
# ---------------- GEMINI CONFIG ----------------
GEMINI_API_KEY = "PASTE_YOUR_API_KEY_HERE"

client = genai.Client(api_key=GEMINI_API_KEY)

def generate_resume_summary(resume_text):
    try:
        prompt = f"""
        You are a professional resume writer.
        Create a concise, impactful resume summary (3–4 lines) based on this resume:

        {resume_text}
        """

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error generating summary: {e}"


# ---------------- HELPERS ----------------
def load_users():
    return json.load(open(USER_DB)) if os.path.exists(USER_DB) else {}

def save_users(users):
    json.dump(users, open(USER_DB,"w"), indent=4)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    if len(password)<8: return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password): return "Password must have an uppercase letter."
    if not re.search(r"\d", password): return "Password must have a number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password): return "Password must have a special char."
    return None

def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        pdf = PyPDF2.PdfReader(file)
        for page in pdf.pages:
            txt = page.extract_text()
            if txt: text += txt + "\n"
    elif file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    return text.strip()

def extract_skills(text, skills_list):
    found = set()
    text_lower = text.lower()
    for skill in skills_list:
        if re.search(r'\b'+re.escape(skill.lower())+r'\b', text_lower):
            found.add(skill)
    return list(found)

def skills_match(resume_text, jd_text, all_skills):
    resume_skills = extract_skills(resume_text, all_skills)
    jd_skills = extract_skills(jd_text, all_skills)
    missing_skills = [s for s in jd_skills if s not in resume_skills]
    matched_skills = [s for s in jd_skills if s in resume_skills]
    if not jd_skills: return 100, []
    score = int((len(matched_skills)/len(jd_skills))*100)
    return score, missing_skills

def semantic_match(resume_text, jd_text, model):
    from sentence_transformers import util
    import re
    resume_sent = [s.strip() for s in re.split(r"[.!\n]", resume_text) if s.strip()]
    jd_sent = [s.strip() for s in re.split(r"[.!\n]", jd_text) if s.strip()]
    if not resume_sent or not jd_sent: return 0
    emb_resume = model.encode(resume_sent, convert_to_tensor=True)
    emb_jd = model.encode(jd_sent, convert_to_tensor=True)
    sim_matrix = util.cos_sim(emb_resume, emb_jd)
    max_sims = sim_matrix.max(dim=0)[0]
    return int(max_sims.mean().item()*100)

def get_verdict_color(score):
    if score>=85: return "✅ Excellent Fit","#28a745"
    elif score>=70: return "👍 Strong Match","#17a2b8"
    elif score>=50: return "⚠️ Potential Fit","#ffc107"
    else: return "❌ Low Match","#dc3545"

def display_dashboard_card(title, skills_score, soft_score, final_score, verdict, color, missing_skills):
    st.markdown(f"""
    <div style="max-width:480px;margin:auto;border:2px solid {color};padding:20px;border-radius:12px;background:#f8f9fa;box-shadow:0 4px 8px rgba(0,0,0,0.1);margin-bottom:20px;">
        <h4 style="color:{color};">{title} - <b>{verdict}</b></h4>
        <div style="display:flex;align-items:center;margin-bottom:10px;">
            <div style="width:100px;height:100px;border:5px solid {color};border-radius:50%;display:flex;justify-content:center;align-items:center;font-size:2.5em;font-weight:bold;color:{color};">{final_score}</div>
            <div style="margin-left:20px;">
                <p style="margin:0;"><b>Skills Match:</b> {skills_score}%</p>
                <div style="background:#e9ecef;border-radius:5px;height:10px;width:150px;">
                    <div style="background:{color};height:100%;width:{skills_score}%;border-radius:5px;"></div>
                </div>
                <p style="margin:10px 0 0;"><b>Soft Match:</b> {soft_score}%</p>
                <div style="background:#e9ecef;border-radius:5px;height:10px;width:150px;">
                    <div style="background:{color};height:100%;width:{soft_score}%;border-radius:5px;"></div>
                </div>
            </div>
        </div>
        <p style="font-weight:bold;">🎯 Missing Skills:</p>
        <p style="color:#6c757d;">{', '.join(missing_skills) if missing_skills else 'None!'}</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="SmartMatch Resume Analyzer", page_icon="📄", layout="wide")

if "role" not in st.session_state: st.session_state["role"]=None
if "authenticated" not in st.session_state: st.session_state["authenticated"]=False
users = load_users()
if "model" not in st.session_state:
    with st.spinner("Loading NLP model..."):
        st.session_state["model"]=SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Persistent Header ----------
# ---------- HEADER WITH LOGOUT INSIDE ----------
col_left, col_right = st.columns([9, 1])

with col_left:
    st.markdown("""
    <div style="
        background:linear-gradient(to right, #e0f7fa, #e1bee7);
        padding:15px 20px;
        border-radius:10px;
        font-size:26px;
        font-weight:bold;
    ">
        SmartMatch
    </div>
    """, unsafe_allow_html=True)

with col_right:
    if st.session_state.get("authenticated"):
        st.markdown("<br>", unsafe_allow_html=True)  # vertical alignment
        if st.button("🚪 Logout"):
            st.session_state["authenticated"] = False
            st.session_state["role"] = None
            st.session_state.pop("user", None)
            st.rerun()



# ---------- Landing Page ----------
if st.session_state["role"] is None:
    st.markdown("<h1 style='text-align:center;'>✨ Smart Resume Matcher</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Analyze resumes against job descriptions instantly.</h3>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1: 
        if st.button("👔 I'm a Recruiter"): st.session_state["role"]="recruiter"; st.rerun()
    with col2: 
        if st.button("🙋 I'm an Applicant"): st.session_state["role"]="applicant"; st.rerun()

# ---------- Authentication ----------
elif not st.session_state["authenticated"]:
    st.markdown(f"<h2 style='text-align:center;'>🔑 {st.session_state['role'].capitalize()} Login</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    auth_mode=st.radio("",["Login","Signup"],horizontal=True)
    email=st.text_input("Email")
    password=st.text_input("Password",type="password")
    if auth_mode=="Signup": confirm_password=st.text_input("Confirm Password",type="password")
    if st.button(auth_mode,use_container_width=True):
        users=load_users()
        if auth_mode=="Login":
            if email not in users: st.error("No account found. Please sign up.")
            elif users[email]["password"]!=hash_password(password): st.error("Invalid password.")
            elif users[email]["role"]!=st.session_state["role"]: st.error(f"This email is registered as {users[email]['role']}.")
            else: st.session_state["authenticated"]=True; st.session_state["user"]=email; st.success("✅ Login successful!"); st.rerun()
        elif auth_mode=="Signup":
            if email in users: st.error("Email exists. Login instead.")
            elif password!=confirm_password: st.error("Passwords do not match.")
            else:
                err=validate_password(password)
                if err: st.error(err)
                else: users[email]={"password":hash_password(password),"role":st.session_state["role"]}; save_users(users); st.success("✅ Account created! Please log in."); st.rerun()

# ---------- Dashboards ----------
else:
    model=st.session_state["model"]
    
    # ---------- Recruiter ----------
    if st.session_state["role"]=="recruiter":
        st.title("📢 Recruiter Dashboard")
        col_jd,col_resumes=st.columns(2)
        with col_jd: jd_files=st.file_uploader("Upload Job Description(s)",type=["pdf","txt"],accept_multiple_files=True)
        with col_resumes: resumes=st.file_uploader("Upload Candidate Resume(s)",type=["pdf","txt"],accept_multiple_files=True)
        if jd_files and resumes:
            results=[]
            for jd in jd_files:
                jd_text=extract_text(jd)
                for resume in resumes:
                    resume_text=extract_text(resume)
                    skills_score,missing_skills=skills_match(resume_text,jd_text,COMMON_SKILLS)
                    soft_score=semantic_match(resume_text,jd_text,model)
                    final_score=int(0.7*skills_score+0.3*soft_score)
                    verdict,color=get_verdict_color(final_score)
                    results.append({
                        "Resume File":resume.name,"JD File":jd.name,
                        "Hard Match":skills_score,"Soft Match":soft_score,
                        "Final Score":final_score,"Verdict":verdict,
                        "Missing Skills":", ".join(missing_skills),
                        "Resume Text":resume_text,"JD Text":jd_text
                    })
            df=pd.DataFrame(results)
            for idx,row in df.iterrows():
                if st.button(f"View Dashboard - {row['Resume File']} / {row['JD File']}",key=f"view_{idx}"):
                    display_dashboard_card(f"{row['Resume File']} vs {row['JD File']}", row['Hard Match'], row['Soft Match'], row['Final Score'], row['Verdict'], get_verdict_color(row['Final Score'])[1], row['Missing Skills'].split(", ") if row['Missing Skills'] else [])
            # Styled table
            def color_verdict(val):
                mapping={"✅ Excellent Fit":"#28a745","👍 Strong Match":"#17a2b8","⚠️ Potential Fit":"#ffc107","❌ Low Match":"#dc3545"}
                return f"background-color:{mapping.get(val,'white')};color:white;text-align:center;"
            st.dataframe(df.drop(columns=["Resume Text","JD Text"]).style.applymap(color_verdict,subset=["Verdict"]),hide_index=True)
    
    # ---------- Applicant ----------
    else:
        st.title("📄 Applicant Dashboard")
        resume_file=st.file_uploader("Upload Your Resume",type=["pdf","txt"])
        job_role=st.selectbox("Select Job Role",list(GENERIC_JDS.keys()))
        if resume_file and job_role:
            resume_text=extract_text(resume_file)
            jd_text=GENERIC_JDS[job_role]
            skills_score,missing_skills=skills_match(resume_text,jd_text,COMMON_SKILLS)
            soft_score=semantic_match(resume_text,jd_text,model)
            # Adjust score for General Purpose
            if job_role=="General Purpose":
                final_score=int(70+0.2*(0.7*skills_score+0.3*soft_score))
            else:
                final_score=int(min(100, (0.7*skills_score+0.3*soft_score)*1.10))
            verdict,color=get_verdict_color(final_score)
            display_dashboard_card(f"{resume_file.name} - {job_role}", skills_score, soft_score, final_score, verdict, color, missing_skills)
            st.info("💡 Analysis shows key missing skills for this role. Adjust your resume accordingly.")
            # ---------- Gemini Resume Summary ----------
            st.markdown("## ✨ AI Resume Summary (Gemini)")

            if st.button("Generate Resume Summary"):
                with st.spinner("Generating AI summary..."):
                    summary = generate_resume_summary(resume_text)
                    st.success("✅ Resume Summary Generated")
                    st.text_area("📌 Suggested Resume Summary", summary, height=180)
