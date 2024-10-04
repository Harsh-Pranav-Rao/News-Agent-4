import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from sentence_transformers import SentenceTransformer
import json
from sklearn.feature_extraction.text import CountVectorizer



# Load SpaCy model and SBERT model
nlp = spacy.load("en_core_web_trf")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load 2-gram word scores
with open('2_gram_words_scores.json', 'r') as file:
    two_gram_word_scores = json.load(file)

with open('word_scores.json', 'r') as file:
    word_scores = json.load(file)



news_sites_categories = {
    "A": [
        ("The Times of India", 5),
        ("Hindustan Times", 5),
        ("The Hindu", 5),
        ("Economic Times", 5),
        ("Business Standard", 5),
        ("Hindustan Times Tech", 5),
        ("India Today Tech", 5),
        ("Financial Express Tech", 5),
        ("News 18 Tech", 5),
        ("south_china_morning_post", 5),
        ("nbc_news", 5)
    ],
    "B": [
        ("CNBC", 4),
        ("Yahoo Finance", 4),
        ("Business Insider", 4),
        ("CNN Business Tech", 4),
        ("Financial Times Tech", 4),
        ("DNA India Tech", 4),
        ("The Hindu Business Line", 4),
        ("indian_express_tech_news", 4),
        ("dealstreet_asia", 4),
        ("mint", 4),
        ("Krasia",4)
    ],
    "C": [
        ("PYMNTS", 3),
        ("TechMeme", 3),
        ("TNW", 3),
        ("Arstechnica", 3),
        ("Readwrite", 3),
        ("Tech in Asia", 3),
        ("Tech Crunch", 3),
        ("Venture Beat", 3),
        ("Tech Crunch Enterprise", 3),
        ("Tech Funding News", 3),
        ("Tech EU", 3),
        ("Macworld", 3),
        ("Nikkei", 3),
        ("The Verge", 3),
        ("Quartz", 3),
        ("Tech Times", 3),
        ("VC Circle", 3),
        ("China Daily", 3),
        ("Zdnet", 3),
        ("the_register", 3)
    ],
    "D": [
        ("EntrackR", 2),
        ("Pandaily", 2),
        ("UK Tech News", 2),
        ("Cloud Computing News", 2),
        ("CGTN Tech", 2),
        ("CGTN business", 2),
        ("Money Control Tech", 2),
        ("Money Control business", 2),
        ("Startup Reporter", 2),
        ("Tech Node", 2),
        ("AsiaTech", 2),
        ("Healthcare Dive", 2),
        ("FirstPost Tech", 2)
    ],
    "E": [
        ("amazon news", 1),
        ("Finextra", 1),
        ("Tech In Africa", 1)
    ]
}


def score_1gram_words(lemmatized_words, word_scores):
    total_score = 0
    for word in lemmatized_words:
        score = word_scores.get(word, 0)
        st.write(f"**{word}** : has a score of {score}")
        total_score += score
    st.write(f":orange[Score of one gram words] : :orange[{total_score}]")
    return total_score

def score_2gram_words(lemmatized_words, two_gram_word_scores):
    total_score = 0
    vectorizer = CountVectorizer(ngram_range=(2, 2))
    lemmatized_text = ' '.join(lemmatized_words)
    ngrams = vectorizer.fit_transform([lemmatized_text])
    feature_names = vectorizer.get_feature_names_out()
    for ngram in feature_names:
        score = two_gram_word_scores.get(ngram, 0)
        st.write(f"{ngram} : has a score of {score}")
        total_score += score
    st.write(f":orange[Score of two gram words] : :orange[{total_score}]")
    return total_score

def get_total_score(similar_titles):
    total_score = 0
    for news_site, title, score in similar_titles:
        for category, sites in news_sites_categories.items():
            for site, value in sites:
                if site == news_site:
                    total_score += value
                    break
    return total_score

def show_site_score(similar_titles):
    for news_site, title, score in similar_titles:
        for category, sites in news_sites_categories.items():
            for site, value in sites:
                if site == news_site:
                    st.write(f"{site} : {value}")
                    break

def preprocess_and_find_similarities(input_title, news_dict, word_scores, two_gram_word_scores, similarity_threshold):
    all_titles = []
    sites = []
    for site, titles in news_dict.items():
        all_titles.extend(titles)
        sites.extend([site] * len(titles))
    
    title_embeddings = model.encode(all_titles)
    
    input_title_lower = input_title.lower()
    doc = nlp(input_title_lower)
    lemmatized_words = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    
    st.subheader(":orange[One Gram Scoring]" ,divider="orange")
    st.write(f"**Lemmatized title**: {' '.join(lemmatized_words)}")
    
    one_gram_score = score_1gram_words(lemmatized_words, word_scores)
    st.write("-----------------------------------------------------------------------------")

    st.subheader(":orange[Two Gram Scoring]" ,divider="orange")
    two_gram_score = score_2gram_words(lemmatized_words, two_gram_word_scores)
    
    
    total_score = one_gram_score + two_gram_score
    st.write("-----------------------------------------------------------------------------")
    
    input_title_embedding = model.encode([input_title])[0]
    similarities = cosine_similarity([input_title_embedding], title_embeddings)[0]
    
    similar_titles = [(sites[idx], all_titles[idx], similarities[idx]) for idx in range(len(similarities)) if similarities[idx] > similarity_threshold]
    similarity_score = get_total_score(similar_titles)
    
    

    result = {
        "one_gram_score": one_gram_score,
        "two_gram_score": two_gram_score,
        "similarity_score": similarity_score,
        "final_score": total_score + similarity_score,
        "similar_titles": similar_titles
    }
    
    return result

def create_news_dict(uploaded_file):
    df = pd.read_excel(uploaded_file)
    titles = df["title"].tolist()

    news_sites = ['PYMNTS', 'TechMeme', 'TNW', 'Arstechnica', 'Readwrite', 'Tech in Asia', 'Tech Crunch', 'EntrackR', 'Venture Beat', 'Pandaily', 'Tech In Africa', 'UK Tech News', 'Cloud Computing News', 'Tech Crunch Enterprise', 'Tech Funding News', 'CGTN Tech', 'CGTN business', 'Money Control Tech', 'Money Control business', 'Economic Times', 'Business Standard', 'Hindustan Times Tech', 'Tech EU', 'Startup Reporter', 'Macworld', 'Tech Node', 'Nikkei', 'Krasia', 'The Verge', 'Yahoo Finance', 'CNBC', 'AsiaTech', 'Quartz', 'Healthcare Dive', 'FirstPost Tech','south_china_morning_post','indian_express_tech_news','dealstreet_asia','nbc_news','the_register','mint','amazon news','Zdnet']

    current_site = None
    news_dict = {}

    for item in titles:
        if item in news_sites:
            current_site = item
            if current_site not in news_dict:
                news_dict[current_site] = []
        else:
            if current_site:
                news_dict[current_site].append(item)

    return news_dict



st.title(':orange[News Agent Scoring v1]')

uploaded_file = st.file_uploader(":orange[Choose an Excel file]", type="xlsx")

if uploaded_file:
    news_dict = create_news_dict(uploaded_file)
    input_title = st.text_input(":orange[Enter a news title:]")
    similarity_threshold = st.slider(":orange[Similarity Threshold]", 0.0, 1.0, 0.75)

    if st.button(":orange[Analyze]"):
        if input_title:
            results = preprocess_and_find_similarities(input_title, news_dict, word_scores, two_gram_word_scores, similarity_threshold)


            st.subheader(":orange[Corroboration/Credibility Analysis]" ,divider="orange")

            st.write(f"**Title Analysis for**: {input_title}")

            st.write("**Similar titles to input title (that are above selected threshold)**")
            
            for site, title, score in results['similar_titles']:
                st.markdown(f"**Site:** {site}  \n**Title:** {title}  \n**Similarity:** {score * 100:.1f}%")

            st.write("-----------------------------------------------------------------------------")
            st.write("**News sites of the similar titles and their scores (based on readership count)**")


            show_site_score(results['similar_titles'])

            st.write(f":orange[Total Title Corroboration/Credibility Score]: :orange[{results['similarity_score']}]")
            st.write("-----------------------------------------------------------------------------")


            st.subheader(":orange[Final Score of Title]" ,divider="orange")
            st.write("Final score : one gram word scores + two gram word scores + corroboration/credibility score ")
            st.write(f"Final score : :orange[{results['one_gram_score']}] + :orange[{results['two_gram_score']}] + :orange[{results['similarity_score']}] ")
            st.write(f"**:orange[Final Score]**: :orange[{results['final_score']:.1f}]")
            st.write("-----------------------------------------------------------------------------")
        else:
            st.write("Please enter a news title.")
