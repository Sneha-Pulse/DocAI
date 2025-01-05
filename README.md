# **Document AI**  
Document AI is an advanced NLP-powered web application built using **Streamlit** to facilitate seamless interaction with text-based documents. It integrates cutting-edge machine learning models and natural language processing (NLP) tools to provide a wide range of features for document analysis, summarization, visualization, and more.  

## **Key Features**  

### 1. **Document Summarization**  
- Extract concise summaries from large text documents using state-of-the-art models like GPT-3.5.  
- Customize the summary length and focus based on user preferences.  

### 2. **Text-to-Speech**  
- Convert document content into natural-sounding speech using `pyttsx3`.  
- Supports accessibility for users who prefer auditory interaction.  

### 3. **Question & Answering (Q&A)**  
- Ask questions directly about the uploaded document and get precise answers using models like **BERT** or **GPT variants**.  
- Multiple model support for flexibility and enhanced accuracy.  

### 4. **Named Entity Recognition (NER)**  
- Analyze entities like people, organizations, and locations within the document using **spaCy**.  
- Visualize these entities with an interactive display.  

### 5. **Word Cloud Generation**  
- Create visually appealing word clouds to highlight key terms in the document.  

### 6. **Document Similarity**  
- Measure similarity between two text documents using **TF-IDF** and **Cosine Similarity**.  

### 7. **Interactive Visualizations**  
- Generate bar plots of term frequencies using **Seaborn** and **Matplotlib**.  
- Explore text data with intuitive visual aids.  

### 8. **Export Options**  
- Export analyzed or summarized content as a **PDF** or **CSV** file.  

### 9. **Accessibility Settings**  
- Customize text size and contrast for enhanced readability.  
- High-contrast mode available for visually impaired users.  

### 10. **Custom Backgrounds**  
- Personalize the app with custom background images for an engaging user interface.  

---

## **Tech Stack**  

### 1. **Frontend**  
- **Streamlit**: For building an intuitive, interactive web interface.  

### 2. **Backend and NLP Models**  
- **OpenAI GPT Models**: For text summarization and advanced Q&A.  
- **Transformers by Hugging Face**: For question answering with BERT and other models.  
- **spaCy**: For entity recognition and text analysis.  
- **Sumy**: For extractive summarization using algorithms like LSA and LexRank.  
- **PyPDF2**: For PDF text extraction.  
- **TfidfVectorizer**: For document similarity measurement.  

### 3. **Visualization**  
- **WordCloud**: For generating word clouds.  
- **Matplotlib** and **Seaborn**: For visualizing term frequency and other text patterns.  

### 4. **Text-to-Speech**  
- **pyttsx3**: For converting text to speech.  

---

## **How to Run the Project**  

1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/<your-username>/DocAI.git  
   cd DocAI  
   ```  

2. **Install Dependencies**:  
   Ensure you have Python 3.8+ installed. Run:  
   ```bash  
   pip install -r requirements.txt  
   ```  

3. **Run the Application**:  
   ```bash  
   streamlit run app.py  
   ```  

4. **Access the App**:  
   Open your web browser and navigate to `http://localhost:8501`.  

---

## **Folder Structure**  

```plaintext  
DocAI/  
├── app.py                # Main Streamlit app script  
├── requirements.txt      # Python dependencies  
├── images/               # Background images and logos  
├── data/                 # Sample PDF files  
├── utils/                # Utility scripts for processing  
└── README.md             # Project documentation  
```  

---

## **Contributions**  
Contributions are welcome! Please feel free to raise issues or submit pull requests.  

---

## **License**  
This project is licensed under the [MIT License](LICENSE).  

---

## **Acknowledgements**  
- OpenAI for GPT-3.5 API.  
- Hugging Face for Transformers.  
- Streamlit for enabling rapid app development.  
- The open-source community for their incredible tools and libraries.  

