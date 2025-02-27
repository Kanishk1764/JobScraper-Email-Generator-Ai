# AI-Powered Job Scraper & Cold Email Generator

## Overview
This project is an intelligent automation tool that scrapes job postings from websites, processes them using AI, and generates personalized cold emails for outreach. It leverages **LangChain, ChromaDB, and Groq's LLaMA-3.1-8B model** to extract job details, match them with relevant portfolio projects, and craft professional emails.

## Features
- **Web Scraping**: Extracts job descriptions from career pages.
- **AI-Powered Processing**: Uses Groq's LLaMA-3.1-8B to analyze job descriptions.
- **Portfolio Matching**: Searches a stored portfolio using ChromaDB to find relevant projects.
- **Cold Email Generation**: Creates a professional, AI-generated email for business outreach.

## Technologies Used
- **Python**
- **LangChain**
- **ChromaDB**
- **Groq LLaMA-3.1-8B**
- **Pandas** (for handling CSV data)

## Installation & Setup

### Prerequisites
Ensure you have Python installed and set up a virtual environment if needed.

### Install Dependencies
```bash
pip install chromadb langchain_community langchain_groq pandas
```

### Run the Script
```bash
python job_scraper_email_generator.py
```

## How It Works
1. **Scrape Job Descriptions**: The script extracts job descriptions from a specified website.
2. **Extract Key Details**: The AI processes role, experience, skills, and description.
3. **Portfolio Matching**: The system searches ChromaDB to find relevant projects.
4. **Generate Cold Emails**: Using Groq's AI model, a personalized email is created for outreach.

## Usage
You can modify the script to target different job boards or adjust the email template to fit your needs.

For any inquiries, reach out via @kanishkmishra402@gmail.com
