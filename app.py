import streamlit as st
import sqlite3
import pandas as pd
import os
import requests
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="QueryWhiz - AI SQL Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Background Image ---
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    page_bg = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.75);
    }}

    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.3);
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Apply it
set_background("static/querywhiz_bg.png")# --- White Text for Main UI (excluding blocks) ---
st.markdown("""
<style>
/* Make only general text white, not tables, charts, or widgets */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
    color: white !important;
}

/* Keep sidebar readable */
[data-testid="stSidebar"] {
    color: #1e293b !important;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar text color fix (make all sidebar text black) ---
st.markdown("""
<style>
/* Sidebar text, headers, labels, and buttons */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #000000 !important;  /* black text */
}

/* Sidebar button styling (optional, matching theme) */
[data-testid="stSidebar"] button {
    background-color: #e2e8f0 !important;
    color: #000000 !important;
    border: 1px solid #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 600;
}
[data-testid="stSidebar"] button:hover {
    background-color: #cbd5e1 !important;
}
</style>
""", unsafe_allow_html=True)


# --- Set white text for main content (not sidebar) ---
st.markdown("""
<style>
/* Make all text inside the main app area white */
[data-testid="stAppViewContainer"] {
    color: white;
}

/* Keep sidebar text dark for contrast */
[data-testid="stSidebar"] {
    color: #1e293b !important;
}

/* Adjust headers, tables, and buttons for visibility */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    color: white !important;
}

/* Dataframe table header and text */
.dataframe thead tr th {
    color: #ffffff !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
}
.dataframe tbody tr td {
    color: #f1f5f9 !important;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
}
div.stButton > button:hover {
    background-color: #1d4ed8;
}
</style>
""", unsafe_allow_html=True)

# --- Final Fix: Dropdown (Select a Plot) visibility ---
st.markdown("""
<style>
/* === SELECT DROPDOWN STYLING (BLACK TEXT) === */

/* Closed dropdown box */
div[data-baseweb="select"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #6366f1 !important;
    border-radius: 8px !important;
}

/* Text inside the closed dropdown */
div[data-baseweb="select"] span,
div[data-baseweb="select"] div,
div[data-baseweb="select"] input {
    color: #000000 !important;
}

/* Open dropdown container */
div[data-baseweb="popover"] {
    background-color: #ffffff !important;
    border: 1px solid #6366f1 !important;
    border-radius: 8px !important;
}

/* Make sure ALL text inside dropdown popover is black */
div[data-baseweb="popover"] * {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Individual options */
div[data-baseweb="option"],
div[data-baseweb="menu-item"],
div[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Hover effect */
div[data-baseweb="option"]:hover,
div[data-baseweb="menu-item"]:hover,
div[role="option"]:hover {
    background-color: #e2e8f0 !important;
    color: #000000 !important;
}

/* Selected option */
div[data-baseweb="option"][aria-selected="true"],
div[data-baseweb="menu-item"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
    background-color: #c7d2fe !important;
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Fix: Black Text Only Inside Light Boxes ---
st.markdown("""
<style>
/* Target only the boxes and light sections */
div[style*="background-color:#f1f5f9;"],
div[style*="background-color: rgb(241, 245, 249)"],
textarea,
.stTextArea textarea,
.stTextInput input,
.stCode pre,
.stCode code,
pre code {
    color: #000000 !important;  /* black text inside light boxes */
}

/* For the AI Explanation block */
div[style*="border-left:5px solid #2563eb;"] p,
div[style*="border-left:5px solid #2563eb;"] span {
    color: #000000 !important;
}

/* Keep everything else white */
[data-testid="stAppViewContainer"] p:not([style*="background-color"]),
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4,
[data-testid="stAppViewContainer"] h5,
[data-testid="stAppViewContainer"] h6,
[data-testid="stAppViewContainer"] label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ✅ Fix: Make the "Generated SQL Query" text visible in st.code() */
.stCode, [data-testid="stCode"] pre, [data-testid="stCode"] code {
    color: #000000 !important;              /* Black text */
    background-color: #f9fafb !important;   /* Light background */
    border-radius: 8px !important;
    padding: 10px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 15px !important;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

/* ✅ Ensure syntax-highlighted spans inside code also turn black */
.stCode span, [data-testid="stCode"] span {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ✅ FINAL OVERRIDE: Force sidebar text color to black */

/* Sidebar entire container */
[data-testid="stSidebar"] * {
    color: #000000 !important;  /* Black text everywhere in sidebar */
}

/* Sidebar headers and labels */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] h5,
[data-testid="stSidebar"] h6,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #000000 !important;
}

/* Sidebar buttons (grey-blue mix as before) */
[data-testid="stSidebar"] div.stButton > button {
    background-color: #607d8b !important;   /* grey-blue mix */
    color: #ffffff !important;              /* white button text */
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: 0.3s ease-in-out;
}

/* Hover effect for sidebar buttons */
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #546e7a !important;   /* darker hover */
    color: #ffffff !important;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)


# --- Custom CSS Styling ---
st.markdown("""
<style>
/* Background & Font */
body {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
h1 {
    color: #2563eb;
    text-align: center;
    margin-bottom: 10px;
}

/* Subheaders */
h2, h3 {
    color: #334155;
    margin-top: 25px;
}

/* Buttons */
div.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    font-weight: 600;
    border: none;
    transition: all 0.2s ease-in-out;
}
div.stButton > button:hover {
    background-color: #1e40af;
    transform: scale(1.02);
}

/* DataFrame */
.dataframe {
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    padding: 10px;
}

/* Success/Warning/Error boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Plot display */
img {
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Button Style (Grey-Blue Mix) ---
st.markdown("""
<style>
/* Sidebar buttons only */
[data-testid="stSidebar"] div.stButton > button {
    background-color: #607d8b !important;   /* grey-blue mix */
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: 0.3s ease-in-out;
}

/* Hover effect for sidebar buttons */
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #546e7a !important;   /* slightly darker grey-blue on hover */
    color: #ffffff !important;
    transform: scale(1.03);
}
</style>
""", unsafe_allow_html=True)

# --- FINAL FIX: AI Explanation Text Color ---
st.markdown("""
<style>
.ai-explanation-box,
.ai-explanation-box p,
.ai-explanation-box span,
.ai-explanation-box div,
.ai-explanation-box * {
    color: #000000 !important;  /* Force black text */
}
.ai-explanation-box p {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for persistent data
if "sql_query" not in st.session_state:
    st.session_state["sql_query"] = ""
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "plot_index" not in st.session_state:
    st.session_state["plot_index"] = 0
if "db_tables" not in st.session_state:
    st.session_state["db_tables"] = []

# Ensure directories exist
os.makedirs("db", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

# Mistral API Configuration
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_AGENT_URL = "https://api.mistral.ai/v1/agents/completions"
MISTRAL_API_KEY = "jDcCdZ0dkkTKEjAfVIoLWbRcmZ4ktwBs"
MISTRAL_AGENT_ID = "ag:79ec7e4f:20250328:untitled-agent:36859aa3"

def load_database_schema(db_path):
    """Loads the schema of the uploaded SQLite database and returns schema and table names."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schema = ""

    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        schema += f"Table: {table_name}\n"
        schema += ", ".join([f"{col[1].lower()} ({col[2]})" for col in columns]) + "\n\n"

    conn.close()
    return schema.strip(), [table[0] for table in tables]

def generate_sql(user_input, db_path):
    """Converts a natural language query to SQL using Mistral API, handling only SELECT queries."""
    table_schema, _ = load_database_schema(db_path)

    prompt = (
        f"You are an expert in writing SQLite queries. "
        f"Based on the following database schema, generate only a valid and complete SQL SELECT query "
        f"compatible with SQLite syntax (use LIMIT instead of TOP, use double quotes for strings if needed). "
        f"Do NOT include explanations, comments, or markdown formatting. "
        f"If the user’s request is unrelated to the schema, respond exactly with 'ERROR: No relevant table found'.\n\n"
        f"Schema:\n{table_schema}\n"
        f"User request: '{user_input}'"
    )


    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-small",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }

    response = requests.post(MISTRAL_URL, headers=headers, data=json.dumps(payload))
    print(f"🟢 Sending request to Mistral with schema:\n{table_schema}")
    print(f"🟢 User Query: {user_input}")
    print(f"🟢 Mistral Response: {response.status_code} - {response.text}")

    if response.status_code == 200:
        result = response.json()["choices"][0]["message"]["content"].strip()
        # Extract SQL query from potential code block
        sql_match = re.search(r'```sql\n(.*?)\n```', result, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            sql_query = result.strip()

        print(f"🟢 Extracted SQL Query: {sql_query}")  # Debug log

        if sql_query == "ERROR: No relevant table found":
            return None
        if sql_query.lower().startswith("select"):
            return sql_query
        return None  # Return None for non-SELECT queries
    else:
        return f"Error: {response.status_code}, {response.text}"

def extract_python_code(response_text):
    """Extracts the Python code block from the Mistral response, ignoring explanatory text."""
    # Look for the code block starting with the first import statement
    code_start = re.search(r'^(import|from)\s', response_text, re.MULTILINE)
    if not code_start:
        return None
    
    code_start_pos = code_start.start()
    # Find the end of the code by looking for explanatory text or the end of the string
    explanatory_text = re.search(r'\n\s*This script|\n\s*Here\'s a', response_text[code_start_pos:], re.MULTILINE)
    if explanatory_text:
        code_end_pos = code_start_pos + explanatory_text.start()
    else:
        code_end_pos = len(response_text)
    
    code = response_text[code_start_pos:code_end_pos].strip()
    return code

def call_mistral_for_visualization(csv_file):
    """Uses Mistral to generate a visualization script based on CSV data."""
    with open(csv_file, "r") as f:
        csv_data = f.read()

    prompt = f"""
    You are an expert Python developer.
    Given the following CSV data, generate only a valid Python script that:
    - Sets Matplotlib to the non-interactive 'Agg' backend BEFORE ANY IMPORTS using:
    ```python
    import matplotlib
    matplotlib.use('Agg')
    ```
    - Includes all necessary imports (e.g., pandas, matplotlib.pyplot).
    - Reads the CSV file 'generated_data.csv' using pd.read_csv('generated_data.csv').
    - Creates plots (e.g., bar, scatter, box, line, word cloud,) based on the data.
    - Saves the plot as 'static/plots/plot1.png', 'static/plots/plot1.png', and so on.
    - Clears old plots from 'static/plots' before generating new ones.
    - Do NOT include any explanations, comments, or markdown formatting outside the code.

    CSV Data:
    {csv_data}
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "agent_id": MISTRAL_AGENT_ID,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(MISTRAL_AGENT_URL, headers=headers, json=payload)
    print(f"🟢 Visualization Request Response: {response.status_code} - {response.text}")

    if response.status_code == 200:
        response_text = response.json()["choices"][0]["message"]["content"].strip()
        # Extract the actual Python code
        code = extract_python_code(response_text)
        if code:
            return code
        else:
            raise Exception("No valid Python code found in Mistral response")
    else:
        raise Exception(f"Mistral API failed: {response.status_code} - {response.text}")

def generate_ai_explanation(df, sql_query):
    """Generate AI-based summary or insights for the SQL query result."""
    if df.empty:
        return "No data available for generating insights."

    # Convert the DataFrame into a readable text form
    data_preview = df.head(10).to_csv(index=False)
    num_rows, num_cols = df.shape

    prompt = f"""
    You are a data analyst AI.
    The following SQL query was executed:
    {sql_query}

    The resulting dataset has {num_rows} rows and {num_cols} columns.
    Here is a preview of the data:
    {data_preview}

    Please write a concise, insightful summary of what this data shows.
    Focus on trends, patterns, or notable values, using plain English.
    """

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }

    response = requests.post(MISTRAL_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"].strip()
        return content
    else:
        return f"Error: {response.status_code}, {response.text}"


# Title
# --- Gradient Header ---
st.markdown("""
<div style="background: #0b1a4a;
            padding: 8px 15px; border-radius: 10px; text-align:center; margin-top: -180px;">
    <h1 style="color:white;">QueryWhiz ✨</h1>
    <p style="color:#e2e8f0; font-size:16px;">Ask questions. Get insights. Visualize data effortlessly.</p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar for QueryWhiz ---
from PIL import Image
import streamlit as st

# --- Sidebar Header ---
st.sidebar.image("static/Title.png", use_container_width=True)
st.sidebar.markdown(
    "<h2 style='text-align:center; color:#2563eb;'>QueryWhiz 🪄</h2>", 
    unsafe_allow_html=True
)
st.sidebar.markdown("<p style='text-align:center; color:#475569;'>Your AI SQL Companion</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# --- Chat History Section ---
st.sidebar.markdown("### 💬 Chat History")

# Initialize chat history in session
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if st.session_state["chat_history"]:
    for i, q in enumerate(reversed(st.session_state["chat_history"][-5:]), 1):
        st.sidebar.markdown(f"**{i}.** {q}")
else:
    st.sidebar.info("No chat history yet.")

# --- Quick Actions ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Quick Actions")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🧹 Clear History"):
        st.session_state["chat_history"].clear()
        st.sidebar.success("Cleared!")
with col2:
    if st.button("🧾 About"):
        st.sidebar.info("""
        **QueryWhiz** helps you:
        - Ask questions in English  
        - Auto-generate SQL queries  
        - Execute & visualize results  
        - Get AI explanations instantly  

        Developed by *Keerthana D* 💙
        """)

# --- Developer Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<p style='text-align:center; font-size:13px; color:#64748b;'>© 2025 QueryWhiz | Powered by Mistral AI</p>",
    unsafe_allow_html=True
)

# Instructions
st.markdown("""
### How to Use:
1. Upload a SQLite database to start.
2. View the tables present in the uploaded database below.
3. Type a natural language query and click "Generate SQL".
4. Click "Execute Query" to see the results.
5. Click "Generate Visualization" to create plots.
6. Use the dropdown or "Next"/"Previous" buttons to view all generated plots.
7. Click "Clear" to reset outputs (database remains loaded). """)

# Database Upload
db_file = st.file_uploader("Upload SQLite Database", type=["db"])
if db_file:
    with open("db/database.db", "wb") as f:
        f.write(db_file.read())
    st.success("Database uploaded successfully!")
    # Load and display tables
    schema, table_names = load_database_schema("db/database.db")
    st.session_state["db_tables"] = table_names
    st.subheader("Tables in Uploaded Database:")
    # Display tables in a tabular format
    df_tables = pd.DataFrame(table_names, columns=["Table Name"])
    st.table(df_tables)

# Natural Language Query Input
user_query = st.text_area("Enter your natural language query")

# Generate SQL Button
if st.button("Generate SQL"):
    if not os.path.exists("db/database.db"):
        st.error("Please upload a database first!")
    else:
        sql_query = generate_sql(user_query, "db/database.db")
        if sql_query:
            st.session_state["sql_query"] = sql_query
            st.session_state["chat_history"].append(user_query)
        else:
            st.session_state["sql_query"] = "ERROR: No relevant table found"

# Display Generated SQL Query (Persistent)
if st.session_state["sql_query"]:
    st.subheader("Generated SQL Query:")
    st.code(st.session_state["sql_query"], language="sql")

# Execute Query Button
if st.button("Execute Query"):
    if not st.session_state["sql_query"] or "ERROR" in st.session_state["sql_query"]:
        st.error("Generate a valid SQL query first!")
    else:
        try:
            conn = sqlite3.connect("db/database.db")
            df = pd.read_sql_query(st.session_state["sql_query"], conn)
            st.session_state["results_df"] = df
            df.to_csv("generated_data.csv", index=False)
            conn.close()
        except Exception as e:
            st.error(f"Query Execution Failed: {str(e)}")

# Display Query Results (Persistent)
if st.session_state["results_df"] is not None:
    st.subheader("Query Results:")
    st.dataframe(st.session_state["results_df"])

    # AI Explanation Button
    if st.button("Generate AI Explanation"):
        with st.spinner("Generating AI insights..."):
            explanation = generate_ai_explanation(
                st.session_state["results_df"],
                st.session_state["sql_query"]
            )
            st.markdown("### 🧠 AI Explanation of Results")
            with st.container():
                st.markdown(f"""
                <div class="ai-explanation-box" style="background-color:#f1f5f9; border-left:5px solid #2563eb; 
                            padding:15px; border-radius:8px; margin-top:10px;">
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)


# Generate Visualization Button
if st.button("Generate Visualization"):
    if not os.path.exists("generated_data.csv"):
        st.error("Execute a query first to generate results!")
    else:
        try:
            # Clear old plots
            for file in os.listdir("static/plots"):
                os.remove(os.path.join("static/plots", file))
            # Generate visualization code
            try:
                generated_code = call_mistral_for_visualization("generated_data.csv")
                # Execute the generated code directly
                exec_globals = {
                    "matplotlib": matplotlib,
                    "plt": plt,
                    "pd": pd,
                    "os": os
                }
                exec(generated_code, exec_globals)
            except Exception as e:
                st.warning(f"Mistral API visualization failed: {str(e)}. Using fallback visualization.")
                # Fallback: Hardcoded visualization
                df = pd.read_csv("generated_data.csv")
                plt.bar(df.iloc[:, 0], df.iloc[:, 1])
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig("static/plots/fallback_plot.png")
                plt.close()

            plots = [f for f in os.listdir("static/plots") if f.endswith(".png")]
            if plots:
                st.success("Visualizations generated successfully!")
            else:
                st.warning("No plots found in static/plots/ after generation.")
        except Exception as e:
            st.error(f"Visualization Generation Failed: {str(e)}")

# Display Plots with Dropdown and Next Button
plots = [f for f in os.listdir("static/plots") if f.endswith(".png")]
if plots:
    st.subheader("Visualizations:")
    
    # Dropdown for plot selection
    plot_options = [f"Plot {i+1}: {plot}" for i, plot in enumerate(plots)]
    selected_plot = st.selectbox("Select a plot", plot_options, index=st.session_state["plot_index"])
    st.session_state["plot_index"] = plot_options.index(selected_plot)
    
    # Display selected plot
    plot_path = os.path.join("static/plots", plots[st.session_state["plot_index"]])
    st.image(plot_path, caption=plots[st.session_state["plot_index"]], use_column_width=True)
    
    # Navigation Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous Plot") and st.session_state["plot_index"] > 0:
            st.session_state["plot_index"] -= 1
            st.rerun()
    with col2:
        if st.button("Next Plot") and st.session_state["plot_index"] < len(plots) - 1:
            st.session_state["plot_index"] += 1
            st.rerun()

# Clear Button
if st.button("Clear"):
    st.session_state["sql_query"] = ""
    st.session_state["results_df"] = None
    st.session_state["plot_index"] = 0
    if os.path.exists("generated_data.csv"):
        os.remove("generated_data.csv")
    for file in os.listdir("static/plots"):
        os.remove(os.path.join("static/plots", file))
    st.success("Cleared all outputs!")
