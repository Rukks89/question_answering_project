import streamlit as st
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)
model = BertForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

def get_answer(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    return answer

st.title("🧠 Question Answering System using BERT")
context = st.text_area("Enter Paragraph:")
question = st.text_input("Enter Question:")

if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Enter both paragraph and question!")
    else:
        answer = get_answer(context, question)
        st.success(answer)
