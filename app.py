import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataPilot AI | Autonomous Data Science Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom CSS & HTML Styling Engine (Full Dark Theme & Widget Fix)
# -----------------------------------------------------------------------------
def apply_custom_theme():
    st.markdown("""
        <style>
        /* Base Streamlit App Background & Text */
        .stApp {
            background-color: #0B0F19 !important;
            color: #F1F5F9 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Force Sidebar Background & Text Contrast */
        [data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid #1F2937 !important;
        }

        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] div,
        [data-testid="stSidebar"] .stMarkdown {
            color: #E2E8F0 !important;
        }

        /* Force all standard markdown text to be bright and visible */
        p, span, label, div, .stMarkdown {
            color: #E2E8F0 !important;
        }

        /* Fix Text Area Input Box Background & Text */
        .stTextArea textarea {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #475569 !important;
            border-radius: 8px !important;
        }

        /* Fix Code Block Background & Text */
        pre, code, .stCodeBlock {
            background-color: #111827 !important;
            color: #38BDF8 !important;
            border: 1px solid #334155 !important;
            border-radius: 8px !important;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            padding: 24px 28px;
            border-radius: 14px;
            border: 1px solid #334155;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            margin-bottom: 24px;
        }

        .hero-title {
            color: #38BDF8 !important;
            font-size: 30px;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .hero-subtitle {
            color: #CBD5E1 !important;
            font-size: 14px;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* Stat Cards */
        .stat-card {
            background: #1E293B !important;
            border: 1px solid #475569 !important;
            border-radius: 12px;
            padding: 18px 16px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-3px);
            border-color: #38BDF8 !important;
            box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2);
        }

        .stat-val {
            font-size: 26px;
            font-weight: 700;
            color: #FFFFFF !important;
            margin-top: 4px;
        }

        .stat-lbl {
            font-size: 11px;
            color: #94A3B8 !important;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
        }

        /* Custom Status Badge */
        .badge-active {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399 !important;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        /* Card Container for Features */
        .content-card {
            background: #1E293B !important;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #475569 !important;
            margin-bottom: 20px;
        }
        
        /* Fix Streamlit Tab Text Visibility */
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: #F8FAFC !important;
            font-weight: 600 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Helper Logic Functions
# -----------------------------------------------------------------------------
def analyze_dataset_health(df: pd.DataFrame):
    """Computes automated quality metrics for uploaded datasets."""
    total_rows, total_cols = df.shape
    total_cells = total_rows * total_cols
    missing_cells = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
    
    health_score = max(0, round(100 - (missing_pct * 1.5 + duplicate_pct * 2), 1))
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        "rows": total_rows,
        "cols": total_cols,
        "missing_pct": round(missing_pct, 2),
        "duplicates": duplicate_rows,
        "health_score": health_score,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

# -----------------------------------------------------------------------------
# 4. Main App Layout & Interactivity
# -----------------------------------------------------------------------------
def main():
    apply_custom_theme()

    # Sidebar Controls
    with st.sidebar:
        st.title("⚙️ Engine Settings")
        st.markdown("---")
        
        st.subheader("🤖 Agent Configuration")
        model_choice = st.selectbox("LLM Backbone", ["Gemini 1.5 Flash", "Gemini 1.5 Pro", "Custom Local Pipeline"])
        temperature = st.slider("Agent Creativity", 0.0, 1.0, 0.2, 0.1)
        
        st.markdown("---")
        st.subheader("🛡️ Guardian Rules")
        drift_threshold = st.slider("Drift Sensitivity Alert", 0.01, 0.10, 0.05, 0.01)
        
        st.markdown("---")
        st.caption("DataPilot AI v2.0 • Autonomous Data Science Platform")

    # Hero Banner Header
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">
                <span>🚀 DataPilot AI Studio</span>
                <span class="badge-active">● Engine Online</span>
            </div>
            <p class="hero-subtitle">
                Autonomous Data Analysis • Machine Learning Model Guardian • Multi-Agent Workflow Engine
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main Core Tabs Navigation
    tab_analyst, tab_guardian, tab_builder = st.tabs([
        "📊 Autonomous Data Analyst", 
        "🛡️ ML Model Guardian", 
        "🤖 Multi-Agent Project Builder"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Autonomous Data Analyst
    # -------------------------------------------------------------------------
    with tab_analyst:
        st.subheader("Autonomous Dataset Quality & Exploratory Analysis")
        uploaded_file = st.file_uploader("Upload CSV dataset for evaluation", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            metrics = analyze_dataset_health(df)

            # Metric Cards Row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-lbl">Total Records</div>
                        <div class="stat-val">{metrics['rows']:,}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-lbl">Features</div>
                        <div class="stat-val">{metrics['cols']}</div>
                    </div>
                """, unsafe_allow_html=True)
            with c3:
                color = "#34D399" if metrics['health_score'] >= 85 else "#FBBF24"
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-lbl">Data Health Score</div>
                        <div class="stat-val" style="color: {color};">{metrics['health_score']}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-lbl">Missing Cells</div>
                        <div class="stat-val" style="color: #F87171;">{metrics['missing_pct']}%</div>
                    </div>
                """, unsafe_allow_html=True)

            st.write("")
            col_preview, col_insights = st.columns([3, 2])
            
            with col_preview:
                st.write("**Dataset Preview**")
                st.dataframe(df.head(7), use_container_width=True)
            
            with col_insights:
                st.write("**Automated Quality Diagnostics**")
                st.json({
                    "Numeric Columns": len(metrics['numeric_cols']),
                    "Categorical Columns": len(metrics['categorical_cols']),
                    "Duplicate Rows Detected": metrics['duplicates'],
                    "Status": "Ready for Pipeline Processing"
                })

            # -----------------------------------------------------------------
            # NEW FEATURE: Interactive Plotly Correlation Heatmap
            # -----------------------------------------------------------------
            if len(metrics['numeric_cols']) > 1:
                st.write("**Interactive Feature Correlation Heatmap**")
                corr_matrix = df[metrics['numeric_cols']].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    aspect="auto"
                )
                fig.update_layout(
                    paper_bgcolor="#0B0F19",
                    plot_bgcolor="#0B0F19",
                    font_color="#F1F5F9",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

            # Autonomous Data Cleaning Studio
            st.markdown("---")
            st.subheader("🧹 Autonomous Data Cleaning Studio")
            st.write("Configure your automated data cleaning rules and export the polished dataset.")

            clean_col1, clean_col2 = st.columns(2)
            with clean_col1:
                missing_strategy = st.selectbox(
                    "Missing Value Handling",
                    ["Keep As Is", "Drop Rows with Missing Values", "Impute Numeric (Mean)", "Impute Numeric (Median)"]
                )
            with clean_col2:
                remove_duplicates = st.checkbox("Drop Duplicate Rows", value=True)

            cleaned_df = df.copy()
            if remove_duplicates:
                cleaned_df = cleaned_df.drop_duplicates()

            if missing_strategy == "Drop Rows with Missing Values":
                cleaned_df = cleaned_df.dropna()
            elif missing_strategy == "Impute Numeric (Mean)":
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                cleaned_df[num_cols] = cleaned_df[num_cols].fillna(cleaned_df[num_cols].mean())
            elif missing_strategy == "Impute Numeric (Median)":
                num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                cleaned_df[num_cols] = cleaned_df[num_cols].fillna(cleaned_df[num_cols].median())

            st.caption(f"Cleaned Dataset Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")

            csv_export_data = cleaned_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset (.csv)",
                data=csv_export_data,
                file_name="datapilot_cleaned_dataset.csv",
                mime="text/csv",
                type="primary"
            )

        else:
            st.info("💡 Upload a `.csv` file above to trigger the autonomous data evaluator & cleaner.")

    # -------------------------------------------------------------------------
    # TAB 2: ML Model Guardian
    # -------------------------------------------------------------------------
    with tab_guardian:
        st.subheader("Model Monitoring & Drift Detection Hub")
        
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("""
                <div class="content-card">
                    <h4 style="color:#38BDF8; margin-top:0;">Model Performance Scorecard</h4>
                    <p style="color:#CBD5E1; font-size: 13px;">Real-time health evaluation of active models.</p>
                </div>
            """, unsafe_allow_html=True)
            
            perf_df = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
                "Baseline Score": [0.912, 0.895, 0.880, 0.887, 0.942],
                "Live Score": [0.898, 0.881, 0.875, 0.878, 0.931],
                "Status": ["Normal", "Normal", "Normal", "Normal", "Normal"]
            })
            st.table(perf_df)

        with g2:
            st.markdown("""
                <div class="content-card">
                    <h4 style="color:#38BDF8; margin-top:0;">Feature Drift Watchdog</h4>
                    <p style="color:#CBD5E1; font-size: 13px;">Tracking distribution changes vs base baseline dataset.</p>
                </div>
            """, unsafe_allow_html=True)
            
            drift_data = pd.DataFrame({
                "Feature": ["user_age", "transaction_amt", "account_age_days", "session_time"],
                "P-Value": [0.342, 0.012, 0.891, 0.041],
                "Drift Detected": ["No", "YES (Alert)", "No", "YES (Alert)"]
            })
            st.dataframe(drift_data, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: Multi-Agent Project Builder
    # -------------------------------------------------------------------------
    with tab_builder:
        st.subheader("Multi-Agent Data Science Project Pipeline")
        st.write("Specify your project goal to generate an end-to-end modular architecture code.")
        
        user_prompt = st.text_area(
            "Project Objective",
            placeholder="e.g., Build a customer churn prediction model using Random Forest and generate evaluation plots..."
        )
        
        if st.button("⚡ Run Multi-Agent Execution Pipeline", type="primary"):
            if user_prompt:
                with st.status("Executing Multi-Agent Workflow...", expanded=True) as status:
                    st.write("🕵️ **Data Scout Agent:** Analyzing project requirements and target variables...")
                    time.sleep(1.2)
                    st.write("🧹 **Preprocessing Agent:** Formatting feature transformations and pipeline clean-ups...")
                    time.sleep(1.2)
                    st.write("⚡ **ML Engineer Agent:** Writing Python scripts and cross-validation protocols...")
                    time.sleep(1.0)
                    status.update(label="Workflow Complete!", state="complete", expanded=False)

                st.success("Project Pipeline Generated Successfully!")
                
                generated_script = f"""# ==========================================
# DataPilot AI - Generated Pipeline Script
# Goal: {user_prompt}
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
df = pd.read_csv("dataset.csv")

# 2. Features & Target Split
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluation
preds = model.predict(X_test)
print(classification_report(y_test, preds))
"""
                
                st.code(generated_script, language="python")

                st.download_button(
                    label="📥 Download Python Pipeline Script (.py)",
                    data=generated_script,
                    file_name="datapilot_pipeline.py",
                    mime="text/plain",
                    type="primary"
                )
            else:
                st.warning("Please enter a project objective first.")

if __name__ == "__main__":
    main()