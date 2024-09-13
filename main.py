import streamlit as st

import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
except ImportError:
    st.error("Failed to import torch or transformers. Please check your installation.")
    st.stop()

import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
from textblob import TextBlob
import matplotlib.pyplot as plt
import PyPDF2
import io

# Initialize UPSC topics with subtopics
UPSC_TOPICS = {
    "Polity and Governance": ["Constitution", "Parliament", "Judiciary", "Local Government"],
    "Economy and Social Development": ["Macroeconomics", "Banking", "Social Welfare", "Poverty"],
    "Science and Technology": ["Space", "Biotechnology", "IT", "Energy"],
    "Environment and Ecology": ["Climate Change", "Biodiversity", "Pollution", "Conservation"],
    "International Relations": ["Foreign Policy", "International Organizations", "Geopolitics"],
    "History and Culture": ["Ancient India", "Medieval India", "Modern India", "Art and Culture"],
    "Geography": ["Physical Geography", "Human Geography", "Economic Geography", "Geopolitics"],
    "Internal Security": ["Terrorism", "Border Management", "Cyber Security", "Insurgency"]
}

# Initialize SQLite database
conn = sqlite3.connect('upsc_news.db')
c = conn.cursor()

# Create tables with proper indexing
c.executescript('''
    CREATE TABLE IF NOT EXISTS articles
    (id INTEGER PRIMARY KEY AUTOINCREMENT,
     date TEXT,
     title TEXT,
     content TEXT,
     topic TEXT,
     subtopic TEXT,
     summary TEXT,
     sentiment REAL,
     url TEXT,
     content_hash TEXT UNIQUE);
    
    CREATE INDEX IF NOT EXISTS idx_date ON articles(date);
    CREATE INDEX IF NOT EXISTS idx_topic ON articles(topic);
    CREATE INDEX IF NOT EXISTS idx_subtopic ON articles(subtopic);
    
    CREATE TABLE IF NOT EXISTS study_progress
    (date TEXT PRIMARY KEY,
     hours REAL,
     streak INTEGER);
''')

# Load pre-trained models
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier_name = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_name).to(device)

    summarizer_name = "facebook/bart-large-cnn"
    summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_name)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_name).to(device)

    return classifier_tokenizer, classifier_model, summarizer_tokenizer, summarizer_model, device

classifier_tokenizer, classifier_model, summarizer_tokenizer, summarizer_model, device = load_models()

# Improved classification function with subtopic prediction
def classify_text(text):
    inputs = classifier_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topic_idx = torch.argmax(probabilities).item()
    topic = list(UPSC_TOPICS.keys())[topic_idx]

    vectorizer = TfidfVectorizer()
    subtopics = UPSC_TOPICS[topic]
    vectorizer.fit(subtopics + [text])
    subtopic_vectors = vectorizer.transform(subtopics)
    text_vector = vectorizer.transform([text])
    similarities = cosine_similarity(text_vector, subtopic_vectors)
    subtopic = subtopics[similarities.argmax()]

    return topic, subtopic

# Improved summarization function
def summarize_text(text, max_length=150):
    inputs = summarizer_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Sentiment analysis function
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to insert article into database
def insert_article(date, title, content, topic, subtopic, summary, sentiment, url):
    content_hash = hashlib.md5(content.encode()).hexdigest()
    try:
        c.execute("INSERT INTO articles (date, title, content, topic, subtopic, summary, sentiment, url, content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (date, title, content, topic, subtopic, summary, sentiment, url, content_hash))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Duplicate article

# Asynchronous web scraping function
async def fetch_article(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('h1').text.strip() if soup.find('h1') else "No title found"
                content = ' '.join([p.text for p in soup.find_all('p') if len(p.text.split()) > 10])
                return title, content, url
            else:
                return None
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None

async def scrape_articles(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Function to log study progress
def log_study_progress(hours):
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT streak FROM study_progress ORDER BY date DESC LIMIT 1")
    result = c.fetchone()
    current_streak = result[0] if result else 0

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    c.execute("SELECT date FROM study_progress WHERE date = ?", (yesterday,))
    if c.fetchone():
        current_streak += 1
    else:
        current_streak = 1

    c.execute("INSERT OR REPLACE INTO study_progress (date, hours, streak) VALUES (?, ?, ?)", (today, hours, current_streak))
    conn.commit()

# Function to get study progress
def get_study_progress():
    c.execute("SELECT date, hours, streak FROM study_progress ORDER BY date")
    return c.fetchall()

# New function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Main Streamlit app
def main():
    st.title("Advanced UPSC Preparation System")

    menu = ["News Analysis", "Study Progress", "Topic Explorer", "PDF Analysis"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "News Analysis":
        st.subheader("News Analysis")

        tab1, tab2 = st.tabs(["Manual Input", "Web Scraping"])

        with tab1:
            date = st.date_input("Article Date", datetime.now())
            title = st.text_input("Article Title")
            content = st.text_area("Article Content", height=200)
            url = st.text_input("Article URL (optional)")

            if st.button("Process Article"):
                if content and title:
                    with st.spinner("Processing article..."):
                        topic, subtopic = classify_text(content)
                        summary = summarize_text(content)
                        sentiment = analyze_sentiment(content)

                        if insert_article(date.strftime("%Y-%m-%d"), title, content, topic, subtopic, summary, sentiment, url):
                            st.success(f"Category: {topic} - {subtopic}")
                            st.subheader("Summary:")
                            st.write(summary)
                            st.subheader("Sentiment:")
                            st.write(f"Sentiment score: {sentiment:.2f}")
                            st.success("Article saved to database!")
                        else:
                            st.warning("This article seems to be a duplicate and was not added to the database.")
                else:
                    st.warning("Please enter both title and content.")

        with tab2:
            urls = st.text_area("Enter URLs (one per line)")

            if st.button("Scrape Articles"):
                if urls:
                    with st.spinner("Scraping articles..."):
                        url_list = urls.split('\n')
                        results = asyncio.run(scrape_articles(url_list))

                        success_count = 0
                        for result in results:
                            if result:
                                title, content, url = result
                                topic, subtopic = classify_text(content)
                                summary = summarize_text(content)
                                sentiment = analyze_sentiment(content)
                                if insert_article(datetime.now().strftime("%Y-%m-%d"), title, content, topic, subtopic, summary, sentiment, url):
                                    success_count += 1

                        st.success(f"Successfully scraped and processed {success_count} out of {len(url_list)} articles!")
                else:
                    st.warning("Please enter at least one URL.")

    elif choice == "Study Progress":
        st.subheader("Study Progress Tracker")

        hours = st.number_input("Hours studied today", min_value=0.0, max_value=24.0, step=0.5)
        if st.button("Log Progress"):
            log_study_progress(hours)
            st.success("Progress logged successfully!")

        progress_data = get_study_progress()
        if progress_data:
            df = pd.DataFrame(progress_data, columns=["Date", "Hours", "Streak"])
            df["Date"] = pd.to_datetime(df["Date"])

            st.subheader("Study History")
            fig = px.line(df, x="Date", y="Hours", title="Daily Study Hours")
            st.plotly_chart(fig)

            st.subheader("Current Streak")
            current_streak = df["Streak"].iloc[-1]
            st.metric("Consecutive Study Days", current_streak)

            st.subheader("Total Study Hours")
            total_hours = df["Hours"].sum()
            st.metric("Total Hours", f"{total_hours:.1f}")

            # IAS progress visualization
            st.subheader("IAS Preparation Progress")
            target_date = datetime(2025, 7, 1)
            start_date = datetime(2024, 9, 13)
            total_days = (target_date - start_date).days
            days_elapsed = (datetime.now() - start_date).days
            progress_percentage = min(days_elapsed / total_days, 1)

            text = "IAS ARTH SINGH"
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, text, fontsize=30, fontweight='bold', ha='center', va='center', alpha=0.1)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axvline(progress_percentage, color='g', linewidth=5)
            ax.axis('off')
            st.pyplot(fig)

            st.write(f"Progress: {progress_percentage:.1%}")
            st.write(f"Days until exam: {total_days - days_elapsed}")

    elif choice == "Topic Explorer":
        st.subheader("UPSC Topic Explorer")

        selected_topic = st.selectbox("Select a topic", list(UPSC_TOPICS.keys()))

        c.execute("SELECT title, summary, sentiment FROM articles WHERE topic=? ORDER BY date DESC LIMIT 10", (selected_topic,))
        articles = c.fetchall()

        if articles:
            st.subheader(f"Recent Articles on {selected_topic}")
            for title, summary, sentiment in articles:
                st.write(f"**{title}**")
                st.write(summary)
                st.write(f"Sentiment: {sentiment:.2f}")
                st.write("---")

    elif choice == "PDF Analysis":
        st.subheader("PDF Document Analysis")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file is not None:
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)

                if pdf_text:
                    topic, subtopic = classify_text(pdf_text)
                    summary = summarize_text(pdf_text)
                    sentiment = analyze_sentiment(pdf_text)

                    st.success(f"Category: {topic} - {subtopic}")
                    st.subheader("Summary:")
                    st.write(summary)
                    st.subheader("Sentiment:")
                    st.write(f"Sentiment score: {sentiment:.2f}")

                    # Option to save to database
                    if st.button("Save to Database"):
                        title = uploaded_file.name
                        if insert_article(datetime.now().strftime("%Y-%m-%d"), title, pdf_text, topic, subtopic, summary, sentiment, ""):
                            st.success("PDF content saved to database!")
                        else:
                            st.warning("This content seems to be a duplicate and was not added to the database.")
                else:
                    st.error("Failed to extract text from the PDF. Please ensure it's a valid PDF file.")

if __name__ == "__main__":
    main()
