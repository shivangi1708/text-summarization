# 📝 Text Summarization with Facebook BART

This project performs **abstractive text summarization** using the **Facebook BART** model from Hugging Face. It includes a complete training pipeline on the SAMSum dataset and a Flask-based web interface for real-time prediction.

---

## 🚀 Features

- 📚 Trained on the [SAMSum dataset](https://huggingface.co/datasets/samsum)
- 🤖 Uses `facebook/bart-base` model from Hugging Face Transformers
- 🧪 Modular training pipeline with stages:
  - Data ingestion
  - Data validation
  - Data transformation
  - Model training
  - Model evaluation
- 🌐 Web interface for user-friendly summarization via Flask
- 🖥️ CPU-friendly setup (no GPU required)

---

## 📸 Demo

Here’s what the summarization web interface looks like:

![Web App Screenshot](assets/sample_output.png)
