Technical PRD: Agri-Credit Helper MVP
Version: 2.0 (Execution Blueprint)
Date: August 12, 2025
Author: Sidharth Rajmohan

1. Overview
This document provides the complete technical specifications for building the Agri-Credit Helper. It details the architecture, data models, logic, and implementation steps required to create a working MVP for the hackathon. The system consists of two main components: a real-time conversational agent for farmers and an offline data ingestion pipeline for administrators.

2. User Stories
As a Farmer, I want to:
Send a voice or text message in my local language to a Telegram bot.
Ask a question about a government financial scheme (e.g., "How much interest do I pay on a KCC loan?").
Receive a clear, accurate answer in my own language, based on official documents.
As an Admin, I want to:
Use a simple web page to upload new government policy PDF files.
Have the system automatically "learn" from these PDFs to keep the agent's knowledge up-to-date.
3. System Architecture
Component A: The Real-Time Agent (Farmer-Facing)

code
Mermaid
graph TD
    subgraph "User on Telegram"
        A(User) -- "Sends Hindi Voice/Text" --> B[Telegram API]
    end
    
    subgraph "Backend on Google Cloud Run"
        B -- "Webhook" --> C{Agent Core}
        C -- "1. Transcribe & Understand Query" --> D[Gemini 1.5 Flash]
        D -- "2. Embed User Query" --> E[Google Embedding API]
        E -- "3. Retrieve Top 25 Docs" --> F[Supabase Vector Store]
        F -- "4. Rerank for Relevance" --> G[Cohere Rerank API]
        G -- "5. Get Top 3 Docs" --> H{Final Context}
        H -- "6. Generate Grounded Answer" --> D
    end

    subgraph "Response"
        D -- "7. Response in Hindi" --> C
        C -- "Sends Message" --> B
        B -- "Delivers to User" --> A
    end
Component B: The Data Ingestion Pipeline (Admin-Facing)

code
Mermaid
graph TD
    subgraph "Admin Interface"
        A[Admin uploads PDF via Streamlit UI] --> B{File sent to Supabase Storage};
    end
    
    subgraph "Serverless Processing (Cloud Function)"
        B -- "Trigger" --> C{Ingestion Logic};
        C -- "1. Extract Layout & Text" --> D["Advanced OCR <br> (e.g., PyMuPDF, Mistral Doc Scanner concept)"];
        D -- "2. Clean Text with AI" --> E["Gemini LLM <br> (Removes page numbers, etc.)"];
        E -- "3. Recursive Chunking" --> F[Text split into small, overlapping chunks];
        F -- "4. Embed Chunks" --> G[Google Embedding API];
        G -- "5. Store in DB" --> H[Supabase Vector Store];
    end```

### 4. Database Schema (Supabase)
The `pg_vector` extension must be enabled.

| Table Name | Column | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **documents** | `id` | `uuid` (PK) | Unique ID for the chunk. |
| | `content` | `text` | The text of the document chunk. |
| | `embedding` | `vector(768)` | Embedding vector. **Note:** Use the correct dimension for the embedding model. |
| | `source_document` | `text` | Filename of the source PDF for citations. |
| | `created_at` | `timestampz` | Ingestion timestamp. |
| **users** | `telegram_user_id` | `bigint` (PK) | The user's unique Telegram ID. |
| | `last_seen` | `timestampz` | Last interaction time. |
| **chat_history**| `id` | `uuid` (PK) | Unique message ID. |
| | `telegram_user_id` | `bigint` (FK) | User who sent the message. |
| | `role` | `text` | "user" or "assistant". |
| | `content` | `text` | The message text. |

### 5. API & Service Definitions
*   **Telegram API:** To send/receive messages and handle voice notes.
*   **Google Gemini API (via AI Studio):** For main reasoning, voice transcription, and document cleaning. Use model `Gemini 2.5-Flash`.
*   **Google Embedding API:** For creating vectors from text. Use model `gemini-embedding-001`.
*   **Cohere Rerank API:** To increase the relevance of retrieved documents before sending them to the LLM.
*   **Supabase:** For Postgres DB, Vector Store, and file storage for the ingestion pipeline.

### 6. Detailed Component Breakdown

#### **Component A: Real-Time Agent Logic (The Bot)**
1.  **Input Handling:**
    *   The bot receives a message object from Telegram.
    *   If it's a voice message, download the `.oga` file. Send the audio file directly to the **Gemini 1.5 Flash** model for transcription. It can handle this natively.
    *   If it's text, use the text directly., tell the model to reframe the query as best it can according to the users intent and for our usecase, of finding the best policy if its relavant to that to query the database in the best way 
2.  **Query Embedding:**
    *   Take the transcribed/text query from the user.
    *   Make an API call to the **Google Embedding API** (`gemini-embedding-001`) to get a vector representation of the query.
3.  **Retrieval:**
    *   Use the query vector to perform a similarity search on the `documents` table in your Supabase vector store.
    *   Retrieve the **Top 25** most similar document chunks. This is your initial, wide net of potentially relevant context.
4.  **Reranking:**
    *   Take the user's original query (string) and the 25 retrieved document chunks (list of strings).
    *   Make a single API call to the **Cohere Rerank API**.
    *   Set the `top_n` parameter to **3**. The API will return the 3 most relevant documents from the initial 25.
5.  **Answer Generation:**
    *   Construct a final prompt for the **Gemini 1.5 Flash** model. This prompt must include:
        *   **System Instructions:** The core persona ("You are Agri-Credit Helper...").
        *   **User's Original Question:** E.g., *"What is the interest rate..."*
        *   **Language Instruction:** A clear instruction: *"The user asked this question in [Language]. You MUST answer in [Language]."*
        *   **The Reranked Context:** The content of the top 3 documents returned by Cohere.
    *   Make the API call to Gemini. The response will be your final, grounded answer.
6.  **Response:** Send the generated text answer back to the user via the Telegram API.

#### **Component B: Data Ingestion Pipeline (The Brain-Loader)**
1.  **Admin UI (Streamlit):**
    *   A simple web page with a file uploader widget.
    *   When an admin uploads a PDF, the app sends the file directly to a designated bucket in **Supabase Storage**.
2.  **Processing Trigger (Cloud Function):**
    *   The file upload to Supabase Storage should trigger a serverless function (e.g., Google Cloud Function). This function performs the following steps:
3.  **Extraction:**
    *   The function downloads the PDF from storage.
    *   It uses a library like **`PyMuPDF`** to extract the text and layout information. This is more robust than simple text extraction. This step conceptually fulfills the role of the "Mistral Doc Scanner."
4.  **AI-Powered Cleaning:**
    *   The raw extracted text is passed to **Gemini** with a specific cleaning prompt: *"You are a document sanitation AI... remove page numbers, headers, footers... output only the cleaned text."*
5.  **Chunking (Splitting):**
    *   Use LangChain's **`RecursiveCharacterTextSplitter`** on the cleaned text.
    *   Start with `chunk_size=1200`, `chunk_overlap=200`. This ensures semantic concepts are not split harshly.
6.  **Embedding & Storing:**
    *   For each text chunk created:
        *   Call the **Google Embedding API** (`gemini-embedding-001`) to get its vector.
        *   Create a record containing the chunk's text content, its vector, and the name of the source PDF.
        *   Insert this record into the `documents` table in Supabase.

### 7. Phased Implementation Plan

1.  **Week 1 (Core MVP - August 18th Deadline):**
    *   **Day 1-2:** Setup all services (Supabase, Google Cloud, Cohere, Telegram). Create the database schema. Write a local `ingest.py` script and manually process 2-3 key government scheme PDFs.
    *   **Day 3-4:** Build the core agent backend on your local machine. Get the RAG flow (Retrieve -> Rerank -> Generate) working with hardcoded questions.
    *   **Day 5:** Connect the backend to the Telegram API. Test end-to-end with text messages.
    *   **Day 6:** Implement the voice-to-text logic. Record the demo video.
    *   **Day 7:** Clean up code, add comments, and prepare the final submission package.

2.  **Week 2 (Polish for Finals Presentation):**
    *   Build the Streamlit Admin UI for file uploads.
    *   Deploy the ingestion logic as a proper Cloud Function triggered by Supabase Storage.
    *   Add conversation history and a feedback mechanism (👍/👎) to the bot.
    *   Refine prompts and chunking strategy based on test results.