import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# page config
st.set_page_config(
    page_title="INX Employee Performance",
    page_icon="📊",
    layout="wide"
)

#css
st.markdown("""
<style>
.main { background-color: #0b0e14; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }

.stButton>button {
    background: linear-gradient(45deg, #1f6feb, #388bfd);
    color: white; border: none; border-radius: 8px;
    padding: 12px; font-weight: bold; width: 100%;
    transition: 0.3s; box-shadow: 0 4px 15px rgba(56, 139, 253, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(56, 139, 253, 0.5);
}
.intro-box {
    background: #161b22; padding: 25px; border-radius: 15px;
    border-left: 5px solid #388bfd; margin-bottom: 25px;
}
.rating-outstanding {
    background: rgba(240, 192, 64, 0.1); border: 2px solid #f0c040;
    color: #f0c040; padding: 25px; border-radius: 12px; text-align: center;
}
.rating-excellent {
    background: rgba(76, 175, 125, 0.1); border: 2px solid #4caf7d;
    color: #4caf7d; padding: 25px; border-radius: 12px; text-align: center;
}
.rating-good {
    background: rgba(224, 92, 92, 0.1); border: 2px solid #e05c5c;
    color: #e05c5c; padding: 25px; border-radius: 12px; text-align: center;
}
div[data-testid="metric-container"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
</style>
""", unsafe_allow_html=True)

# 43 feature columns after dimensionality reduction.
FEATURE_COLS = [
    'Age', 'DistanceFromHome', 'EmpEducationLevel', 'EmpEnvironmentSatisfaction',
    'EmpHourlyRate', 'EmpJobInvolvement', 'EmpJobLevel', 'EmpJobSatisfaction',
    'NumCompaniesWorked', 'OverTime', 'EmpLastSalaryHikePercent',
    'EmpRelationshipSatisfaction', 'TotalWorkExperienceInYears',
    'TrainingTimesLastYear', 'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany',
    'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
    'Attrition', 'EducationBackground_Marketing', 'EducationBackground_Medical',
    'EducationBackground_Other', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'EmpDepartment_Development', 'EmpDepartment_Finance',
    'EmpDepartment_Research & Development', 'EmpDepartment_Sales',
    'EmpJobRole_Data Scientist', 'EmpJobRole_Delivery Manager',
    'EmpJobRole_Developer', 'EmpJobRole_Finance Manager',
    'EmpJobRole_Healthcare Representative', 'EmpJobRole_Laboratory Technician',
    'EmpJobRole_Manager', 'EmpJobRole_Manufacturing Director',
    'EmpJobRole_Sales Executive', 'EmpJobRole_Sales Representative',
    'EmpJobRole_Senior Developer', 'EmpJobRole_Technical Lead',
    'BusinessTravelFrequency_Travel_Frequently',
    'BusinessTravelFrequency_Travel_Rarely'
]

#loading saved models.
@st.cache_resource
def load_models():
    rf_model = joblib.load('Models/rf_model.pkl')
    scaler   = joblib.load('Models/scaler.pkl')
    return rf_model, scaler

model, scaler = load_models()

#sidebar.
with st.sidebar:
    st.title("📊 INX Performance")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("", ["🏠 Home", "🔮 Predict", "📈 Model Insights", "🏢 Department Analysis"])
    st.markdown("---")
    st.markdown("### Model Info")
    st.write("**Algorithm:** Random Forest")
    st.write("**Tuning:** GridSearchCV")
    st.write("**Accuracy:** 93.33%")
    st.write("**Macro F1:** 0.88")
    st.write("Imbalance Handling", "class_weight='balanced'")
    st.markdown("---")
    st.info("Status: Model Loaded ✅")

# Home Page
if page == "🏠 Home":
    st.title("INX Future Inc — Employee Performance Analysis")

    st.markdown("""
    <div class="intro-box">
        <h3>The Problem</h3>
        INX Future Inc has been experiencing a decline in employee performance indexes
        over recent years. Client satisfaction dropped by 8 percentage points, and
        management needed a data-driven way to understand why — and predict who is at risk
        before it becomes a bigger problem.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Project Objectives")
    st.markdown("""
    - Evaluate and compare performance indexes across departments
    - Identify the top 3 factors affecting employee performance
    - Build a machine learning model to predict employee performance ratings
    - Provide actionable recommendations for INX management
    """)

    st.markdown("---")
    st.markdown("### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Employees", "1,200")
    c2.metric("Total Features", "28")
    c3.metric("Performance Ratings", "2, 3, 4")
    c4.metric("Majority Class", "Rating 3")

    st.markdown("---")
    st.markdown("### Best Model — Random Forest with GridSearchCV")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "93.33%")
    m2.metric("Macro F1-Score", "0.88")

    st.markdown("""
    > Random Forest outperformed all five other models tested — Logistic Regression,
    > SVM, KNN, XGBoost, and ANN — on both accuracy and macro F1-score.
    """)

#prediction.
elif page == "🔮 Predict":
    st.title("🔮 Predict Employee Performance Rating")
    st.markdown("""
    <div class="intro-box">
        Fill in the employee details below and click <strong>Predict Performance</strong>
        to get a rating prediction from the Random Forest model.
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### 👤 Personal Details")
        age           = st.slider("Age", 18, 60, 30)
        education_bg  = st.selectbox("Education Background", [
            "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
        ])
        edu_level     = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                      help="1=Below College, 5=Doctor")
        marital       = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        gender        = st.selectbox("Gender", ["Male", "Female"])
        distance      = st.slider("Distance From Home (km)", 1, 30, 5)
        overtime      = st.selectbox("Works Overtime?", ["No", "Yes"])
        attrition     = st.selectbox("Attrition", ["No", "Yes"])
        travel        = st.selectbox("Business Travel Frequency", [
            "Non-Travel", "Travel_Rarely", "Travel_Frequently"
        ])
        num_companies = st.number_input("Companies Worked Before", 0, 9, 1)

    with right:
        st.markdown("#### 💼 Job & Experience Details")
        department    = st.selectbox("Department", [
            "Sales", "Development", "Research & Development",
            "Human Resources", "Finance", "Data Science"
        ])
        job_role      = st.selectbox("Job Role", [
            "Data Scientist", "Delivery Manager", "Developer",
            "Finance Manager", "Healthcare Representative",
            "Laboratory Technician", "Manager", "Manufacturing Director",
            "Sales Executive", "Sales Representative",
            "Senior Developer", "Technical Lead",
            "Research Scientist", "Human Resources",
            "Technical Architect", "Manager R&D",
            "Senior Manager R&D", "Research Director"
        ])
        job_level     = st.selectbox("Job Level", [1, 2, 3, 4, 5],
                                      help="1=Entry, 5=Executive")
        job_involve   = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                      help="1=Low, 4=Very High")
        job_sat       = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                      help="1=Low, 4=Very High")
        env_sat       = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                      help="1=Low, 4=Very High")
        rel_sat       = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4],
                                      help="1=Low, 4=Very High")
        wlb           = st.selectbox("Work Life Balance", [1, 2, 3, 4],
                                      help="1=Bad, 4=Best")
        hourly_rate   = st.slider("Hourly Rate", 30, 100, 60)
        salary_hike   = st.slider("Last Salary Hike (%)", 11, 25, 14)
        training      = st.number_input("Training Times Last Year", 0, 6, 2)
        total_exp     = st.slider("Total Work Experience (years)", 0, 40, 5)
        exp_company   = st.slider("Experience at INX (years)", 0, 40, 3)
        exp_role      = st.slider("Years in Current Role", 0, 18, 2)
        yrs_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
        yrs_manager   = st.slider("Years With Current Manager", 0, 17, 2)

    st.markdown("---")

    if st.button("PREDICT PERFORMANCE"):
        #all 43 features set.
        input_dict = {col: 0 for col in FEATURE_COLS}

        # numerical features
        input_dict['Age']                          = age
        input_dict['DistanceFromHome']             = distance
        input_dict['EmpEducationLevel']            = edu_level
        input_dict['EmpEnvironmentSatisfaction']   = env_sat
        input_dict['EmpHourlyRate']                = hourly_rate
        input_dict['EmpJobInvolvement']            = job_involve
        input_dict['EmpJobLevel']                  = job_level
        input_dict['EmpJobSatisfaction']           = job_sat
        input_dict['NumCompaniesWorked']           = num_companies
        input_dict['OverTime']                     = 1 if overtime == "Yes" else 0
        input_dict['EmpLastSalaryHikePercent']     = salary_hike
        input_dict['EmpRelationshipSatisfaction']  = rel_sat
        input_dict['TotalWorkExperienceInYears']   = total_exp
        input_dict['TrainingTimesLastYear']        = training
        input_dict['EmpWorkLifeBalance']           = wlb
        input_dict['ExperienceYearsAtThisCompany'] = exp_company
        input_dict['ExperienceYearsInCurrentRole'] = exp_role
        input_dict['YearsSinceLastPromotion']      = yrs_promotion
        input_dict['YearsWithCurrManager']         = yrs_manager
        input_dict['Attrition']                    = 1 if attrition == "Yes" else 0

        # education background — Life Sciences and Technical Degree stay 0 (noise columns)
        if education_bg == "Marketing":
            input_dict['EducationBackground_Marketing'] = 1
        elif education_bg == "Medical":
            input_dict['EducationBackground_Medical'] = 1
        elif education_bg == "Other":
            input_dict['EducationBackground_Other'] = 1

        # marital status — Divorced stays 0 (base category from drop_first)
        if marital == "Married":
            input_dict['MaritalStatus_Married'] = 1
        elif marital == "Single":
            input_dict['MaritalStatus_Single'] = 1

        # department — Data Science and Human Resources stay 0 (noise columns)
        if department == "Development":
            input_dict['EmpDepartment_Development'] = 1
        elif department == "Finance":
            input_dict['EmpDepartment_Finance'] = 1
        elif department == "Research & Development":
            input_dict['EmpDepartment_Research & Development'] = 1
        elif department == "Sales":
            input_dict['EmpDepartment_Sales'] = 1

        # job role — roles that were noise columns stay 0
        role_map = {
            'Data Scientist':            'EmpJobRole_Data Scientist',
            'Delivery Manager':          'EmpJobRole_Delivery Manager',
            'Developer':                 'EmpJobRole_Developer',
            'Finance Manager':           'EmpJobRole_Finance Manager',
            'Healthcare Representative': 'EmpJobRole_Healthcare Representative',
            'Laboratory Technician':     'EmpJobRole_Laboratory Technician',
            'Manager':                   'EmpJobRole_Manager',
            'Manufacturing Director':    'EmpJobRole_Manufacturing Director',
            'Sales Executive':           'EmpJobRole_Sales Executive',
            'Sales Representative':      'EmpJobRole_Sales Representative',
            'Senior Developer':          'EmpJobRole_Senior Developer',
            'Technical Lead':            'EmpJobRole_Technical Lead',
        }
        if job_role in role_map:
            input_dict[role_map[job_role]] = 1

        # travel — Non-Travel stays 0 (base category from drop_first)
        if travel == "Travel_Frequently":
            input_dict['BusinessTravelFrequency_Travel_Frequently'] = 1
        elif travel == "Travel_Rarely":
            input_dict['BusinessTravelFrequency_Travel_Rarely'] = 1

        # scale and predict
        input_df     = pd.DataFrame([input_dict])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)
        pred         = model.predict(input_scaled)[0]
        proba        = model.predict_proba(input_scaled)[0]
        conf         = round(max(proba) * 100, 1)

        st.markdown("### Prediction Result")

        if pred == 4:
            st.markdown(f"""
            <div class="rating-outstanding">
                <h2>🌟 Outstanding (Rating 4)</h2>
                <p style="font-size:18px;">Confidence: <strong>{conf}%</strong></p>
                <hr style="border-color:#444; margin:16px 0;">
                <p>This is a top performer. Prioritise retention, recognition,
                and career progression to keep them engaged.</p>
            </div>
            """, unsafe_allow_html=True)

        elif pred == 3:
            st.markdown(f"""
            <div class="rating-excellent">
                <h2>✅ Excellent (Rating 3)</h2>
                <p style="font-size:18px;">Confidence: <strong>{conf}%</strong></p>
                <hr style="border-color:#444; margin:16px 0;">
                <p>This employee is performing at the expected level.
                Standard engagement and development initiatives apply.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="rating-good">
                <h2>⚠️ Good (Rating 2)</h2>
                <p style="font-size:18px;">Confidence: <strong>{conf}%</strong></p>
                <hr style="border-color:#444; margin:16px 0;">
                <p>This employee is underperforming. Consider reviewing their
                environment satisfaction and checking how long they have gone
                without a promotion.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Confidence Breakdown")
        classes = model.classes_
        conf_df = pd.DataFrame({
            'Rating': [f'Rating {c}' for c in classes],
            'Confidence (%)': [round(p * 100, 1) for p in proba]
        })
        st.dataframe(conf_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Insights":
    import matplotlib.pyplot as plt

    st.title("📈 Model Performance Insights")
    st.markdown("""
    <div class="intro-box">
        Comparison of all six models trained during this project.
        The macro F1-score is the most reliable metric here since it treats
        all three performance classes equally.
    </div>
    """, unsafe_allow_html=True)

    models   = ['Logistic Regression', 'SVM', 'KNN', 'ANN', 'XGBoost', 'Random Forest']
    accuracy = [0.7750, 0.8208, 0.7667, 0.8167, 0.9167, 0.9333]
    macro_f1 = [0.68,   0.67,   0.50,   0.71,   0.86,   0.88  ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0b0e14')

    for ax, values, title, color in zip(
        axes,
        [accuracy, macro_f1],
        ['Accuracy Score', 'Macro F1-Score'],
        ['steelblue', 'coral']
    ):
        ax.set_facecolor('#161b22')
        bars = ax.barh(models, values, color=color)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Score', color='white')
        ax.set_title(f'Model Comparison: {title}', color='white',
                     fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', va='center', color='white', fontsize=9)


    st.markdown("---")
    st.markdown("### Key Takeaway")
    st.markdown("""
    Random Forest leads on both accuracy (93.33%) and macro F1 (0.88), confirming
    it as the strongest model across all three classes. KNN's macro F1 of 0.50
    reveals that its accuracy was driven by rating 3 alone. The switch from SMOTE
    to cost-sensitive learning for Random Forest was the key decision that cleaned
    up false positives and improved minority class detection.
    """)

# department analysis
elif page == "🏢 Department Analysis":
    import matplotlib.pyplot as plt

    st.title("🏢 Department Performance Analysis")
    st.markdown("---")

    st.markdown("### Average Performance Rating by Department")

    dept_data = {
        'Department':             ['Finance', 'Sales', 'HR', 'R&D', 'Data Science', 'Development'],
        'Avg Performance Rating': [2.77,       2.86,    2.96,  2.98,  3.00,           3.08        ]
    }
    dept_df = pd.DataFrame(dept_data)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0b0e14')
    ax.set_facecolor('#161b22')
    colors = ['#e05c5c' if v < 2.9 else '#4caf7d' for v in dept_df['Avg Performance Rating']]
    bars = ax.barh(dept_df['Department'], dept_df['Avg Performance Rating'], color=colors)
    ax.set_xlabel('Average Performance Rating', color='white')
    ax.set_title('Department Average Performance Ranking', color='white',
                 fontsize=12, fontweight='bold', pad=12)
    ax.tick_params(colors='white')
    ax.set_xlim(0, 4)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    for bar, val in zip(bars, dept_df['Avg Performance Rating']):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    Departments in **red** fall below 2.9 — Finance (2.77) and Sales (2.86) are
    the two underperforming departments. Development sits at 3.08, making it the
    top-performing department at INX.
    """)

    st.markdown("---")
    st.markdown("### Top 3 Factors Affecting Employee Performance")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("#### 1. Environment Satisfaction")
        st.markdown("""
        **Correlation: +0.41**

        The strongest signal in the dataset. As environment satisfaction rises,
        the number of low performers drops sharply. Employees who feel good about
        where they work consistently score higher ratings.
        """)

    with f2:
        st.markdown("#### 2. Last Salary Hike %")
        st.markdown("""
        **Correlation: +0.36**

        Rating 4 employees receive hikes of 20–22% compared to 14% for ratings
        2 and 3. Pay recognition and performance clearly move together in this
        dataset.
        """)

    with f3:
        st.markdown("#### 3. Years Since Last Promotion")
        st.markdown("""
        **Correlation: -0.17**

        The strongest negative signal. The longer an employee goes without a
        promotion, the lower their performance rating tends to be. Career
        stagnation is a real and measurable drag on output at INX.
        """)

    st.markdown("---")
    st.markdown("### Business Recommendations")
    st.markdown("""
    1. **Invest in the work environment in Finance and Sales** — environment satisfaction
    is the single biggest driver of performance, and these two departments have the
    lowest average ratings.

    2. **Review salary hike structures** — top performers receive 20-22% hikes while
    struggling employees sit at 14%. INX should check whether Finance and Sales
    employees are receiving comparable increases.

    3. **Address promotion bottlenecks** — going years without a promotion hurts
    performance. INX should identify employees stuck in the same role for extended
    periods, particularly in Finance and Sales, and act on it.
    """)

st.markdown("---")
st.caption("INX Future Inc Employee Performance Analysis | Developed by Risper Ngugi | © 2025")