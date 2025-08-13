import streamlit as st
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime

# Import our custom modules
from src.file_handler import FileHandler
from src.text_extractor import TextExtractor
from src.summarizer import TextSummarizer
from src.keyword_extractor import KeywordExtractor
from src.concept_mapper import ConceptMapper
from src.concept_definition_extractor import ConceptDefinitionExtractor


# Page configuration
st.set_page_config(
    page_title="NeuroNotes - AI-Powered Note Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3em;
    color: #2E86AB;
    text-align: center;
    margin-bottom: 1em;
}
.sub-header {
    font-size: 1.2em;
    color: #A23B72;
    text-align: center;
    margin-bottom: 2em;
}
.success-box {
    padding: 1em;
    border-radius: 0.5em;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.info-box {
    padding: 1em;
    border-radius: 0.5em;
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    color: #0c5460;
}
.warning-box {
    padding: 1em;
    border-radius: 0.5em;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 NeuroNotes</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transform your notes into intelligent insights with AI-powered analysis</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Summarization settings
        st.subheader("Summarization")
        summary_method = st.selectbox(
            "Method",
            ["hybrid", "transformer", "extractive"],
            help="Hybrid combines multiple approaches for best results"
        )
        
        summary_length = st.selectbox(
            "Length",
            ["short", "medium", "long"],
            index=1
        )
        
        # Keyword extraction settings
        st.subheader("Keyword Extraction")
        keyword_method = st.selectbox(
            "Method",
            ["all", "keybert", "yake", "rake", "spacy"],
            help="'all' combines multiple algorithms for comprehensive results"
        )
        
        num_keywords = st.slider(
            "Number of keywords",
            min_value=5,
            max_value=30,
            value=15
        )
        # NEW: Concept mapping source selection
        st.subheader("Concept Mapping")
        concept_source = st.radio(
            "Generate concept map from:",
            ["Summary (Recommended)", "Full Document", "Both"],
            help="Summary-based maps are usually cleaner and more focused"
        )
        map_type = st.radio("Concept Map Type", ["Keyword Map","Definition Map"], index=1)
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            show_processing_details = st.checkbox("Show processing details", value=False)
            enable_quiz_generation = st.checkbox("Generate quiz questions", value=False)
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Your Document")
        
        # File upload
        file_handler = FileHandler()
        upload_result = file_handler.upload_file()
        
        if upload_result:
            filename, file_type, uploaded_file = upload_result
            
            # Display file info
            st.success(f"✅ File uploaded: {filename}")
            st.info(f"📄 Type: {file_type.upper()}")
            
            # Process button
            if st.button("🚀 Analyze Document", type="primary"):
                with st.spinner("Processing your document..."):
                    process_document(
                        uploaded_file, filename, file_type,
                        summary_method, summary_length,
                        keyword_method, num_keywords,
                        show_processing_details
                    )
        
        # Display processing status
        if st.session_state.processed_data:
            st.markdown("---")
            st.header("📊 Analysis Complete")
            
            data = st.session_state.processed_data
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Words Extracted", data.get('word_count', 0))
                st.metric("Keywords Found", len(data.get('keywords', [])))
            
            with col_b:
                st.metric("Summary Length", len(data.get('summary', '').split()))
                st.metric("Concepts Mapped", len(data.get('concept_map_data', {}).get('nodes', [])))
    
    with col2:
        if st.session_state.processed_data:
            display_results(st.session_state.processed_data, enable_quiz_generation)
        else:
            st.markdown("""
            <div class="info-box">
                <h3>🎯 How NeuroNotes Works</h3>
                <ol>
                    <li><strong>Upload</strong> your PDF, image, or text file</li>
                    <li><strong>Extract</strong> text using advanced OCR and parsing</li>
                    <li><strong>Summarize</strong> content with AI-powered algorithms</li>
                    <li><strong>Identify</strong> key concepts and keywords</li>
                    <li><strong>Visualize</strong> relationships in interactive concept maps</li>
                    <li><strong>Export</strong> results for future reference</li>
                </ol>
                
                <h4>🔧 Supported File Types:</h4>
                <ul>
                    <li>📄 PDF documents</li>
                    <li>🖼️ Images (PNG, JPG, JPEG, BMP, TIFF)</li>
                    <li>📝 Text files (TXT, MD, RTF)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def process_document(uploaded_file, filename, file_type, summary_method, 
                    summary_length, keyword_method, num_keywords, show_details):
    """Process the uploaded document through the entire pipeline"""
    
    try:
        # Initialize components
        file_handler = FileHandler()
        text_extractor = TextExtractor()
        summarizer = TextSummarizer()
        keyword_extractor = KeywordExtractor()
        concept_mapper = ConceptMapper()
        
        # Steps 1-3: Same as before (save file, extract text, generate summary)
        if show_details:
            st.write("⏳ Saving uploaded file...")
        temp_path = file_handler.save_temp_file(uploaded_file, filename)
        
        if show_details:
            st.write("⏳ Extracting text from document...")
        extracted_text = text_extractor.extract_text(temp_path, file_type)
        
        if not extracted_text.strip():
            st.error("❌ Could not extract text from the document.")
            return
            extractor = ConceptDefinitionExtractor()
            concept_mapper = ConceptMapper()
            if map_type == "Definition Map":
                definitions = extractor.extract_definitions(extracted_text)  # From raw text!
                concept_map_path = concept_mapper.create_definition_map(definitions, filename)
                # Store definitions for sidebar or export
                st.session_state.processed_data = {'definitions': definitions, 'concept_map_path': concept_map_path,}

        
        word_count = len(extracted_text.split())
        if show_details:
            st.write(f"✅ Extracted {word_count} words")
        
        if show_details:
            st.write("⏳ Generating AI summary...")
        summary_result = summarizer.summarize_text(extracted_text, summary_method, summary_length)
        
        # NEW APPROACH: Extract keywords from SUMMARY instead of full text
        if show_details:
            st.write("⏳ Extracting keywords from summary...")
        
        # Use the summary for keyword extraction and concept mapping
        summary_text = summary_result['summary']
        keyword_results = keyword_extractor.extract_keywords(
            summary_text,  # <- Using summary instead of extracted_text
            keyword_method, 
            num_keywords
        )
        
        # Get the best keywords from summary
        if "combined" in keyword_results:
            keywords = keyword_results["combined"]
        else:
            keywords = list(keyword_results.values())[0]
        
        # Find relationships within the summary (more coherent!)
        if show_details:
            st.write("⏳ Analyzing concept relationships in summary...")
        keyword_list = [kw for kw, _ in keywords]
        relationships = keyword_extractor.get_keyword_relationships(
            summary_text,  # <- Using summary for relationships too
            keyword_list
        )
        
        # Create concept map from summary-based concepts
        if show_details:
            st.write("⏳ Creating concept map from summary...")
        concept_map_path = concept_mapper.create_concept_map(keywords, relationships, filename)
        concept_map_data = concept_mapper.export_graph_data()
        central_concepts = concept_mapper.get_central_concepts()
        
        # Store results
        st.session_state.processed_data = {
            'filename': filename,
            'extracted_text': extracted_text,
            'word_count': word_count,
            'summary': summary_result['summary'],
            'summary_info': summary_result,
            'keywords': keywords,
            'all_keywords': keyword_results,
            'relationships': relationships,
            'concept_map_path': concept_map_path,
            'concept_map_data': concept_map_data,
            'central_concepts': central_concepts,
            'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'concept_source': 'summary'  # Track that we used summary
        }
        
        # Cleanup and rerun
        file_handler.cleanup_temp_file(temp_path)
        if show_details:
            st.success("✅ Summary-based concept map created!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ An error occurred during processing: {str(e)}")


def display_results(data, enable_quiz_generation):
    """Display the analysis results"""
    
    st.header("📋 Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Summary", "🔑 Keywords", "🕸️ Concept Map", "📊 Analytics", "💾 Export"])
    
    with tab1:
        st.subheader("AI-Generated Summary")
        
        # Summary info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Method", data['summary_info']['method'].title())
        with col2:
            st.metric("Words", data['summary_info']['word_count'])
        with col3:
            st.metric("Compression", f"{data['summary_info']['compression_ratio']:.1f}x")
        
        # Display summary
        st.markdown(f"""
        <div class="success-box">
        <h4>Summary:</h4>
        {data['summary']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show original text in expander
        with st.expander("📖 View Original Text"):
            st.text_area("Original Content", data['extracted_text'], height=300, disabled=True)
    
    with tab2:
        st.subheader("Extracted Keywords")
        
        # Display keywords in different formats
        display_format = st.radio("Display Format", ["Table", "Cloud", "List"], horizontal=True)
        
        if display_format == "Table":
            # Create DataFrame for keywords
            keywords_df = pd.DataFrame(data['keywords'], columns=['Keyword', 'Score'])
            keywords_df['Rank'] = range(1, len(keywords_df) + 1)
            keywords_df = keywords_df[['Rank', 'Keyword', 'Score']]
            
            st.dataframe(keywords_df, use_container_width=True)
        
        elif display_format == "List":
            for i, (keyword, score) in enumerate(data['keywords'], 1):
                st.write(f"{i}. **{keyword}** (Score: {score:.3f})")
        
        else:  # Cloud format
            st.info("💡 Keyword cloud would be displayed here with interactive visualization")
        
        # Show different extraction methods
        if len(data['all_keywords']) > 1:
            with st.expander("🔍 View Results by Method"):
                for method, keywords in data['all_keywords'].items():
                    if method != 'combined':
                        st.write(f"**{method.upper()}:**")
                        for keyword, score in keywords[:5]:
                            st.write(f"  • {keyword} ({score:.3f})")
    
    with tab3:
        with tab3:
            st.subheader("Definition-Based Concept Map")
            if Path(data['concept_map_path']).exists():
                with open(data['concept_map_path'],'r',encoding='utf-8') as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=570, scrolling=True)
                if 'definitions' in data and data['definitions']:
                    st.markdown("### 📚 Definitions")
                    for c,d in data['definitions'].items():
                        st.markdown(f"- **{c}**: <span style='color:#555;'>{d}</span>", unsafe_allow_html=True)

        
        st.subheader("Interactive Concept Map")
        
        # Display concept map
        if Path(data['concept_map_path']).exists():
            # Read and display HTML file
            with open(data['concept_map_path'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=600, scrolling=True)
            
            # Central concepts
            st.subheader("🎯 Most Central Concepts")
            for i, (concept, centrality) in enumerate(data['central_concepts'], 1):
                st.write(f"{i}. **{concept}** (Centrality: {centrality:.3f})")
        
        else:
            st.error("Concept map file not found. Please regenerate the analysis.")
    
    with tab4:
        st.subheader("Document Analytics")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Words", data['word_count'])
            st.metric("Unique Keywords", len(set([kw for kw, _ in data['keywords']])))
            st.metric("Average Keyword Score", f"{sum([score for _, score in data['keywords']]) / len(data['keywords']):.3f}")
        
        with col2:
            map_stats = data['concept_map_data']['statistics']
            st.metric("Concept Nodes", map_stats['node_count'])
            st.metric("Relationships", map_stats['edge_count'])
            st.metric("Network Density", f"{map_stats['density']:.3f}")
        
        # Keyword relationships
        st.subheader("🔗 Keyword Relationships")
        
        relationships_data = []
        for keyword, related in data['relationships'].items():
            if related:  # Only show keywords that have relationships
                relationships_data.append({
                    'Keyword': keyword,
                    'Related Concepts': ', '.join(related[:3]),  # Show top 3
                    'Connection Count': len(related)
                })
        
        if relationships_data:
            relationships_df = pd.DataFrame(relationships_data)
            st.dataframe(relationships_df, use_container_width=True)
        else:
            st.info("No significant relationships detected between keywords.")
    
    with tab5:
        st.subheader("Export Options")
        
        # Create export data
        export_data = {
            'document': data['filename'],
            'processing_time': data['processing_time'],
            'summary': data['summary'],
            'keywords': [{'keyword': kw, 'score': score} for kw, score in data['keywords']],
            'central_concepts': [{'concept': concept, 'centrality': cent} for concept, cent in data['central_concepts']],
            'statistics': {
                'word_count': data['word_count'],
                'summary_compression': data['summary_info']['compression_ratio'],
                'keyword_count': len(data['keywords'])
            }
        }
        
        # JSON export
        json_str = json.dumps(export_data, indent=2)
        st.download_button(
            label="📥 Download Analysis (JSON)",
            data=json_str,
            file_name=f"neuronotes_analysis_{data['filename']}.json",
            mime="application/json"
        )
        
        # CSV export for keywords
        keywords_df = pd.DataFrame(data['keywords'], columns=['Keyword', 'Score'])
        csv_str = keywords_df.to_csv(index=False)
        st.download_button(
            label="📊 Download Keywords (CSV)",
            data=csv_str,
            file_name=f"keywords_{data['filename']}.csv",
            mime="text/csv"
        )
        
        # Concept map download
        if Path(data['concept_map_path']).exists():
            with open(data['concept_map_path'], 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="🕸️ Download Concept Map (HTML)",
                data=html_content,
                file_name=f"concept_map_{data['filename']}.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main()
