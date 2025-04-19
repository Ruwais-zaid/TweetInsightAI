# TweetInsightAI: Real-time Twitter Sentiment Analysis using DistilBERT

## Project Overview

**TweetInsightAI** is a real-time sentiment analysis tool that utilizes the **DistilBERT** transformer model to classify tweets as **Positive**, **Negative**, or **Neutral**. The app provides an interactive interface built with **Streamlit** to input tweets and receive instant sentiment analysis along with a confidence percentage.

This project uses the **Hugging Face Transformers** library and **DistilBERT** model to perform text classification efficiently, making it suitable for real-time analysis.

![TweetInsightAI Example](https://res.cloudinary.com/ddfmbzizr/image/upload/v1745090323/Screenshot_2025-04-18_125429_iuv1xe.png) 

---

## Key Features:
- Real-time analysis of tweets for sentiment classification.
- Built with **DistilBERT** for fast and accurate predictions.
- **Streamlit** interface for easy interaction.
- **SQLite3** integration to store predictions with timestamps and confidence scores.
- Can classify tweets into three categories: **Positive**, **Negative**, and **Neutral**.

---

## Requirements:
Before running the app, ensure you have the following dependencies:

```bash
pip install -r requirements.txt
