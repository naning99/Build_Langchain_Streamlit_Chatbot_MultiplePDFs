# 1. Reference: 
https://www.youtube.com/watch?v=dXxQ0LR-3Hg

# 2. Deliver
(UI image, not deploy yet)
<img width="1431" alt="image" src="https://github.com/user-attachments/assets/ebf14d9a-1495-4778-aab6-dd71e19840a6">

# 3. Code Structure Design
- Prepare Vectore Store
  - Receive PDF from User;
  - PDFReader to text;
  - Textsplitter to chunks;
  - Embedding Model (OpenAIEmbeddings) & Save to Vector Store (FAISS);
- Define Conversation Chain
  - Components: llm, retriever, memory
- Handler User Input
  - Everytime receive a text input, invoke chain to obtain response
  - Handle memory: add response into chat_history, make it as memory_key, take use of st.session_state
 
# 4. Key Words
- Langchain
- Streamlit
- PDF chatbot
- PdfReader
- FAISS
- Streamlit Memory
  
