# QueryWhiz 💫

**QueryWhiz** is an interactive AI-powered web application that converts natural language queries into executable SQL commands, runs them on an uploaded SQLite database, visualizes the results, and even provides AI-driven explanations.
Built with **Streamlit** and powered by the **Mistral API**, QueryWhiz makes database exploration seamless — no SQL expertise required.

---

## 🚀 Features

* **🧠 Natural Language to SQL Conversion**
  Ask questions in plain English like “Show top 5 customers by total sales” — QueryWhiz automatically generates the SQL query.

* **💾 Database Upload**
  Upload your own `.db` file and instantly view all tables in a clean, tabular layout.

* **⚙️ Query Execution**
  Execute generated SQL queries safely and view results directly in the app.

* **📊 Visualization Generator**
  Automatically generate clear Matplotlib visualizations for your SQL results, with navigation and dropdown support to browse multiple plots.

* **🧩 AI Explanation of Results**
  Get smart, concise, AI-generated insights about your query results — trends, patterns, and key takeaways.

* **🎨 Modern UI**
  A visually appealing Streamlit interface with a gradient title bar, sidebar chat history, and background image support.

* **🧹 Clear Functionality**
  Reset outputs (queries, tables, and plots) while keeping your uploaded database active.

---

## 🧰 Tech Stack

| Component                 | Technology                         |
| ------------------------- | ---------------------------------- |
| **Frontend & Backend**    | [Streamlit](https://streamlit.io/) |
| **AI Model**              | [Mistral API](https://mistral.ai/) |
| **Visualization**         | Matplotlib                         |
| **Database**              | SQLite                             |
| **Languages / Libraries** | Python, Pandas, Requests, Pillow   |

---

## ⚙️ Installation

### Prerequisites

* Python 3.8 or above
* Git (for cloning the repository)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your-username>/QueryWhiz.git
   cd QueryWhiz
   ```

2. **Install Dependencies**

   ```bash
   python -m venv venv # Activate the virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Prepare Directory Structure**
   Make sure these folders exist (the app will create them if missing):

   ```
   db/              → stores uploaded SQLite databases
   static/          → contains background and title images
   static/plots/    → stores generated plots
   ```

4. **Run the Application**

   ```bash
   streamlit run app.py
   ```

   Then open your browser to
   👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🪄 How to Use

1. **Upload** your SQLite database file (`.db`).
2. **View** all tables automatically displayed in a table view.
3. **Ask** any question in plain English.
4. **Generate SQL** and verify the query.
5. **Execute** the query to see results.
6. **Generate Visualization** to view plots.
7. **Get AI Explanation** for instant analytical insights.
8. **Clear** outputs anytime to start fresh.

---

## 🧑‍💻 Developed By

**Keerthana D**
*MSc AI | REVA University*
✨ Passionate about building intelligent, user-friendly data tools.

---

## 📜 License

This project is open source and available under the **MIT License**.
