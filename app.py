import streamlit as st
import transformers
import wikipedia
from transformers import pipeline, Pipeline


def load_wiki_summary(query):

    results = wikipedia.search(query)
    summary = wikipedia.summary(results[0], sentences = 10)
    return summary


def load_qa_pipeline():
    qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
    return qa_pipeline

def answer_question(pipeline, question, paragraph):
    input = {
        'question': question,
        'context': paragraph
    }

    output = pipeline(input)

    return output

def main():

    st.title("Demo wikipedia app")
    st.write("Search topic, ask questions, get answers")

    topic = st.text_input("SEARCH TOPIC", "")
    article_paragraph = st.empty()
    question = st.text_input("QUESTION", "")

    if topic:
        summary = load_wiki_summary(topic)
        article_paragraph.markdown(summary)

    if question != "":

        qa_pipeline = load_qa_pipeline()
        result = answer_question(qa_pipeline, question, summary)
        answer = result['answer']

        st.write(answer)
        





if __name__ == '__main__':
    main()

