so this is my final synopsis document :Capital One Launchpad Hackathon: Synopsis

Team Details
Team Name: DUM-DUM
Team Members:
Sidharth Rajmohan
Theme Details
Theme Name: : Exploring and Building Agentic AI Solutions for a High-Impact Area of Society: Agriculture
Theme Benefits:
This theme directly addresses a critical bottleneck in the Indian agricultural sector: the information gap. Farmers, the backbone of the nation's economy, often face complex challenges that require synthesizing information from diverse domains like finance, policy, and crop science. The true benefit of an agentic AI approach is its potential to democratize access to this vital information. An AI agent can serve as a tireless, multilingual advisor, breaking down complex government policies and financial jargon into simple, actionable advice. By leveraging AI, we can create solutions that overcome traditional barriers of literacy and digital access, directly empowering farmers to make more informed, profitable, and sustainable decisions, thereby supporting not just the individual but the entire agricultural ecosystem.
Synopsis
Solution Overview:
Agri-Credit Helper is an AI-powered conversational agent, accessible via Telegram, designed to demystify agricultural finance for Indian farmers. Our solution directly tackles the challenge of farmers being unable to navigate complex, jargon-filled government policy documents for crucial financial schemes like the Kisan Credit Card (KCC) or PM-KISAN.
A farmer can simply send a text or voice message in their natural, local language (e.g., Hindi, Tamil, or "Hinglish") asking a question like, "I am a wheat farmer in Punjab with 2 acres of land. What loan schemes am I eligible for?" The agent will understand the query, retrieve relevant information from a curated and verified knowledge base of official policy documents, and provide a clear, concise, and fact-based answer in the user's language, citing the source of the information. The goal is to provide a reliable, human-aligned AI advisor that bridges the information gap and fosters financial inclusion.
Technical Stack:
User Interface: Telegram Bot API
Backend & Orchestration: Python, LangChain
AI & Reasoning: Google Gemini Family (specifically, a state-of-the-art model like Gemini 2.5-Flash for advanced reasoning and multilingual capabilities) and Google's Text Embedding models.
Database & Knowledge Base: Supabase (utilizing Postgres for relational data like user profiles/chat history, pg_vector for the vector store, and Supabase Storage for raw document hosting).
Hosting: Google Cloud Run for scalable, serverless deployment of the main agent.
Data Ingestion UI: A separate, simple web application built with Streamlit.
Open Source Software: We will be leveraging key open-source libraries including LangChain, Streamlit, and the pg_vector extension for PostgreSQL.
Decision Rationale:
Our technology choices are guided by the principles of rapid development, scalability, and maintainability—critical for a hackathon MVP and beyond.
Supabase: Chosen over a combination of separate services (like Pinecone + a traditional DB) for its integrated, all-in-one backend. This drastically simplifies development by providing a database, vector store, file storage, and authentication in a single, cohesive platform.
Google Cloud Run: Selected for its serverless container architecture. It allows us to deploy our Python application without managing servers, automatically scales with demand (even to zero, saving costs), and is perfectly suited for our stateless agent design.
LangChain: Employed as the core AI orchestration framework to accelerate development. It provides robust, pre-built components for creating the Retrieval-Augmented Generation (RAG) pipeline, connecting our LLM to the Supabase vector store, and managing conversation history, saving significant boilerplate coding time.
Telegram: Chosen as the primary user interface due to its massive user base in India, its excellent support for both text and voice messages (crucial for low-literacy users), and its simple, developer-friendly API.

Innovation Highlights:
Our solution's innovation lies in its practical application of agentic AI to solve a real-world, high-impact problem.
Conversational Policy Navigation: We are transforming static, unreadable PDF documents into a dynamic, conversational knowledge base. This allows farmers to "talk" to policy documents, a paradigm shift from the current state of information retrieval.
End-to-End Multilingual Voice Interface: The agent is designed from the ground up to handle a user's natural, colloquial voice notes in their native language, process the query, and respond in that same language, making it accessible to the widest possible demographic.
Human-in-the-Loop Maintainability: We are building a separate, user-friendly ingestion application. This allows non-technical administrators to easily update the agent's knowledge base by simply dragging and dropping new policy PDFs, ensuring the system remains current and scalable over time. The ingestion pipeline itself uses an AI-powered cleaning step to ensure data quality.
Direct Focus on Financial Inclusion: By targeting the complex world of agricultural finance, our solution directly aligns with the hackathon's special consideration for innovative financial applications, aiming to empower farmers to secure the credit they are entitled to.

Feasibility and User-Friendliness:
Feasibility: The project is highly feasible within the hackathon timeframe. Our MVP will focus on a core set of 2-3 of the most impactful national schemes (e.g., KCC, PM-KISAN). The use of mature technologies like Supabase, Cloud Run, and LangChain de-risks the implementation and allows us to focus on the quality of the user experience and the accuracy of the RAG pipeline.
User-Friendliness: Our solution is designed for extreme user-friendliness. By building on Telegram, we are meeting users on a platform they already know and trust. The voice-first interaction model is intentionally designed to bypass literacy barriers. The agent's core purpose is to distill complex information into simple, direct, and actionable answers, avoiding the overwhelming detail that plagues the source documents.

Success Metrics:
To evaluate the effectiveness of our agent, we will track a combination of user-centric, performance, and scalability metrics:
User Satisfaction & Engagement:
Daily/Monthly Active Users.
User Retention Rate.
Qualitative Feedback: A simple "👍/👎" feedback mechanism on each answer to gauge its helpfulness.

Performance & Reliability:
Response Latency: Time from query to response.
Groundedness/Hallucination Rate: Internal metric to measure how often the agent provides information not backed by the source documents.
Retrieval Accuracy: Internal metric evaluating if the RAG pipeline retrieved the correct document chunks for a given query.

Scalability & Maintainability:
Time required to add a new policy document to the knowledge base via our ingestion UI.

Methodology/Architecture Diagram
Our solution consists of two core architectural components: the real-time agent that serves the user and the backend pipeline for maintaining its knowledge base.
Figure 1: Main Agent Interaction Flow
Caption: This diagram illustrates the primary user journey. A farmer sends a query in their local language via Telegram. The request is processed by the agent backend on Google Cloud Run, where LangChain orchestrates a Retrieval-Augmented Generation (RAG) process. The Gemini model uses our Supabase vector store to find factual information before formulating and sending a grounded, accurate response back to the user.

Figure 2: Data Ingestion & Maintenance Pipeline

Caption: This diagram showcases the system's maintainability. An administrator uses a simple Streamlit web interface to upload new government policy PDFs. This triggers a serverless function that uses an AI-powered process to clean, chunk, and embed the document's content, seamlessly updating the agent's core knowledge base without any downtime. This ensures the agent's information is always current.

mermaid chart 1 : graph TD
subgraph "Phase 1: Admin Upload"
A(Admin User) -- "Uploads Policy PDF" --> B["Streamlit Web UI"]
end

code
Code

download

content_copy

expand_less
subgraph "Phase 2: Cloud Storage & Trigger"
    B -- "Sends file to" --> C["Supabase Storage <br> (Raw PDF Bucket)"]
    C -- "Triggers Serverless Function" --> D{{Ingestion Script <br> (Hosted on Cloud Run/Function)}}
end

subgraph "Phase 3: AI-Powered Processing Logic"
    D -- "Step 1: Load & Extract" --> E["Raw Text Extracted from PDF"]
    E -- "Step 2: Sanitize & Clean with AI" --> F["Cleaned Policy Text <br> (Headers, footers, page numbers removed by Gemini)"]
    F -- "Step 3: Smart Chunking" --> G["Text split into small, <br> overlapping chunks"]
    G -- "Step 4: Vectorize" --> H["Text Chunks + Vector Embeddings <br> (Generated via Google Embedding Model)"]
end

subgraph "Phase 4: Storing in Knowledge Base"
    H -- "Step 5: Write to Database" --> I["Supabase Vector Store <br> (pg_vector)"]
end,mermaid chart 2: graph TD
subgraph "Phase 1: Admin Upload"
    A(Admin User) -- "Uploads Policy PDF" --> B["Streamlit Web UI"]
end

subgraph "Phase 2: Cloud Storage & Trigger"
    B -- "Sends file to" --> C["Supabase Storage <br> (Raw PDF Bucket)"]
    C -- "Triggers Serverless Function" --> D{{"Ingestion Script <br> (Hosted on Cloud Run/Function)"}}
end

subgraph "Phase 3: AI-Powered Processing Logic"
    D -- "Step 1: Load & Extract" --> E["Raw Text Extracted from PDF"]
    E -- "Step 2: Sanitize & Clean with AI" --> F["Cleaned Policy Text <br> (Headers, footers, page numbers removed by Gemini)"]
    F -- "Step 3: Smart Chunking" --> G["Text split into small, <br> overlapping chunks"]
    G -- "Step 4: Vectorize" --> H["Text Chunks + Vector Embeddings <br> (Generated via Google Embedding Model)"]
end

subgraph "Phase 4: Storing in Knowledge Base"
    H -- "Step 5: Write to Database" --> I["Supabase Vector Store <br> (pg_vector)"]
end
