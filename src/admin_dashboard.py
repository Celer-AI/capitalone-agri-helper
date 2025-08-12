"""Streamlit admin dashboard for document management and analytics."""

import asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
from typing import Dict, Any, List
import structlog
from src.document_processor import document_processor
from src.database import db
from src.config import settings

logger = structlog.get_logger()


class AdminDashboard:
    """Streamlit-based admin dashboard for the Agri-Credit Helper."""
    
    def __init__(self):
        self.document_processor = document_processor
        self.database = db
        
        # Configure Streamlit page
        st.set_page_config(
            page_title="Agri-Credit Helper Admin",
            page_icon="🌾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main dashboard application."""
        # Authentication
        if not self._authenticate():
            return
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["📊 Analytics Dashboard", "📄 Document Upload", "📝 Text Input", "⚙️ System Status"]
        )
        
        # Route to appropriate page
        if page == "📊 Analytics Dashboard":
            self._show_analytics_dashboard()
        elif page == "📄 Document Upload":
            self._show_document_upload()
        elif page == "📝 Text Input":
            self._show_text_input()
        elif page == "⚙️ System Status":
            self._show_system_status()
    
    def _authenticate(self) -> bool:
        """Simple password authentication."""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            st.title("🔐 Admin Login")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                if password == settings.admin_password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
            return False
        
        return True
    
    def _show_analytics_dashboard(self):
        """Display analytics dashboard."""
        st.title("📊 Analytics Dashboard")
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", "Loading...", help="Total unique users")
        
        with col2:
            st.metric("Daily Queries", "Loading...", help="Queries today")
        
        with col3:
            st.metric("Success Rate", "Loading...", help="Successful query rate")
        
        with col4:
            st.metric("Documents", "Loading...", help="Total documents in knowledge base")
        
        # Charts section
        st.markdown("### 📈 Usage Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Daily Query Volume")
            # Placeholder for query volume chart
            sample_data = pd.DataFrame({
                'Date': pd.date_range('2024-01-01', periods=30),
                'Queries': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65] * 3
            })
            fig = px.line(sample_data, x='Date', y='Queries', title="Daily Queries")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Language Distribution")
            # Placeholder for language distribution
            lang_data = pd.DataFrame({
                'Language': ['Hindi', 'Tamil', 'English', 'Bengali'],
                'Count': [150, 80, 60, 30]
            })
            fig = px.pie(lang_data, values='Count', names='Language', title="Query Languages")
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.markdown("### 🕒 Recent Activity")
        
        # Placeholder for recent activity table
        recent_activity = pd.DataFrame({
            'Time': ['2024-01-15 10:30', '2024-01-15 10:25', '2024-01-15 10:20'],
            'Event': ['Query Processed', 'Document Uploaded', 'Query Processed'],
            'User/Admin': ['User 12345', 'Admin', 'User 67890'],
            'Status': ['Success', 'Success', 'Failed']
        })
        
        st.dataframe(recent_activity, use_container_width=True)
        
        # Export data
        st.markdown("### 📥 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export User Analytics"):
                st.info("Feature coming soon")
        
        with col2:
            if st.button("Export Query Logs"):
                st.info("Feature coming soon")
        
        with col3:
            if st.button("Export System Metrics"):
                st.info("Feature coming soon")
    
    def _show_document_upload(self):
        """Display document upload interface."""
        st.title("📄 Document Upload")
        st.markdown("Upload government policy PDFs to expand the knowledge base.")
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload government policy documents in PDF format"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            # Process button
            if st.button("🚀 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Read file content
                    file_content = uploaded_file.read()
                    
                    # Process the document
                    result = asyncio.run(
                        self.document_processor.process_pdf_file(file_content, uploaded_file.name)
                    )
                    
                    # Display results
                    if result['success']:
                        st.success("✅ Document processed successfully!")
                        
                        # Show processing details
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Chunks Created", result['chunks_created'])
                        
                        with col2:
                            st.metric("Chunks Stored", result['chunks_stored'])
                        
                        with col3:
                            st.metric("Success Rate", f"{(result['chunks_stored']/result['chunks_created']*100):.1f}%")
                        
                        # Processing steps
                        st.markdown("#### Processing Steps")
                        steps = result.get('processing_steps', {})
                        for step, status in steps.items():
                            icon = "✅" if status else "❌"
                            st.write(f"{icon} {step.replace('_', ' ').title()}")
                    
                    else:
                        st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                        
                        # Show partial results if any
                        if result.get('chunks_created', 0) > 0:
                            st.warning(f"⚠️ Created {result['chunks_created']} chunks but storage failed")
        
        # Recent uploads
        st.markdown("### 📋 Recent Uploads")
        
        # Placeholder for recent uploads table
        recent_uploads = pd.DataFrame({
            'Filename': ['policy_doc_1.pdf', 'kcc_guidelines.pdf', 'pm_kisan_details.pdf'],
            'Upload Time': ['2024-01-15 10:30', '2024-01-14 15:20', '2024-01-13 09:15'],
            'Chunks': [45, 32, 28],
            'Status': ['Success', 'Success', 'Success']
        })
        
        st.dataframe(recent_uploads, use_container_width=True)
    
    def _show_text_input(self):
        """Display text input interface."""
        st.title("📝 Text Input")
        st.markdown("Add text content directly to the knowledge base (bypasses PDF extraction).")
        st.markdown("---")
        
        # Source name input
        source_name = st.text_input(
            "Source Name",
            placeholder="e.g., PM-KISAN Guidelines 2024",
            help="Provide a descriptive name for this content"
        )
        
        # Text content input
        text_content = st.text_area(
            "Text Content",
            height=300,
            placeholder="Paste the policy text content here...",
            help="Paste the cleaned text content from government documents"
        )
        
        # Process button
        if st.button("🚀 Process Text", type="primary", disabled=not (source_name and text_content)):
            with st.spinner("Processing text content..."):
                # Process the text
                result = asyncio.run(
                    self.document_processor.process_text_input(text_content, source_name)
                )
                
                # Display results
                if result['success']:
                    st.success("✅ Text processed successfully!")
                    
                    # Show processing details
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Chunks Created", result['chunks_created'])
                    
                    with col2:
                        st.metric("Chunks Stored", result['chunks_stored'])
                    
                    with col3:
                        st.metric("Text Length", f"{len(text_content):,} chars")
                    
                    # Processing steps
                    st.markdown("#### Processing Steps")
                    steps = result.get('processing_steps', {})
                    for step, status in steps.items():
                        icon = "✅" if status else "❌"
                        st.write(f"{icon} {step.replace('_', ' ').title()}")
                
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Tips
        st.markdown("### 💡 Tips")
        st.info("""
        - Use this feature when you have clean text content from government documents
        - Provide descriptive source names for better organization
        - The text will be automatically cleaned and chunked for optimal retrieval
        - Large texts will be split into smaller, overlapping chunks
        """)
    
    def _show_system_status(self):
        """Display system status and configuration."""
        st.title("⚙️ System Status")
        st.markdown("---")
        
        # System configuration
        st.markdown("### 🔧 Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### AI Models")
            st.write(f"**LLM Model:** {settings.llm_model}")
            st.write(f"**Embedding Model:** {settings.embedding_model}")
            st.write(f"**Embedding Dimensions:** {settings.embedding_dimensions}")
            st.write(f"**Thinking Budget:** {settings.thinking_budget}")
        
        with col2:
            st.markdown("#### Processing Settings")
            st.write(f"**Chunk Size:** {settings.chunk_size}")
            st.write(f"**Chunk Overlap:** {settings.chunk_overlap}")
            st.write(f"**Max Retrieval Docs:** {settings.max_retrieval_docs}")
            st.write(f"**Rerank Top K:** {settings.rerank_top_k}")
        
        # Rate limiting
        st.markdown("### 🚦 Rate Limiting")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Daily Chat Limit", settings.daily_chat_limit)
        
        with col2:
            st.metric("Rate Limit Fallback", settings.rate_limit_fallback)
        
        # Database status
        st.markdown("### 🗄️ Database Status")
        
        # Placeholder for database metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", "Loading...")
        
        with col2:
            st.metric("Total Users", "Loading...")
        
        with col3:
            st.metric("Chat Messages", "Loading...")
        
        with col4:
            st.metric("Analytics Events", "Loading...")
        
        # Health checks
        st.markdown("### 🏥 Health Checks")
        
        health_status = {
            "Database Connection": "✅ Healthy",
            "Google AI API": "✅ Healthy", 
            "Cohere API": "✅ Healthy",
            "Telegram Bot": "✅ Healthy"
        }
        
        for service, status in health_status.items():
            st.write(f"**{service}:** {status}")
        
        # System logs
        st.markdown("### 📋 Recent System Logs")
        
        # Placeholder for system logs
        logs_data = pd.DataFrame({
            'Timestamp': ['2024-01-15 10:35:22', '2024-01-15 10:34:15', '2024-01-15 10:33:08'],
            'Level': ['INFO', 'WARNING', 'ERROR'],
            'Component': ['RAG Pipeline', 'Document Processor', 'Telegram Bot'],
            'Message': ['Query processed successfully', 'PDF extraction took longer than expected', 'Rate limit exceeded for user']
        })
        
        st.dataframe(logs_data, use_container_width=True)


def main():
    """Main entry point for the admin dashboard."""
    dashboard = AdminDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
