import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import gradio as gr

# Load Env
load_dotenv()

# Load data
books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + '&file=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail'],
)

# Load books description into vertor database
raw_documents = TextLoader("tagged_description.txt", encoding='utf-8').load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

db_books = Chroma.from_documents(
    documents,
    embedding=huggingface_embeddings
)

# Retrieve book by similar query + filter by category/ tone
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    # Retrieve similarity book base on description
    recs = db_books.similarity_search(query, k=initial_top_k)
    # Get book info by isbn (appear in tagged_description)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    # Filter by category
    if category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    # Filter by tone
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

# Return list recommend books
def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_desc_split = ' '.join(truncated_desc_split[:30]) + '...'

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f'{authors_split[0]} and {authors_split[1]}'
        elif len(authors_split) > 2:
            authors_str = f'{", ".join(authors_split[:-1])} and {authors_split[-1]}'
        else:
            authors_str = row['authors']

        caption = f'{row["title"]} by {authors_str}: {truncated_desc_split}'
        results.append((row['large_thumbnail'], caption))

    return results


# Create UI using gradio
categories = ['All'] + sorted(books['simple_categories'].unique())
tones  = ['All'] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g: A story about forgiveness")

        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value='All')
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value='All')
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)

    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)


if __name__ == "__main__":
    dashboard.launch()