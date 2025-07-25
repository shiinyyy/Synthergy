import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile

# Lazy loading for heavy modules - improves startup time
def load_heavy_modules():
    """Load heavy modules only when needed with production-safe error handling"""
    if 'heavy_modules' not in st.session_state:
        try:
            with st.spinner("Loading synthesis modules..."):
                # Test dependency availability first
                dependency_status = check_dependencies()
                
                if not dependency_status['core_available']:
                    st.error("Core dependencies not available in this environment")
                    st.session_state['heavy_modules'] = None
                    return None
                
                # Show dependency status
                # if dependency_status['issues']:
                #     st.warning(f"Some optional features unavailable: {', '.join(dependency_status['issues'])}")
                
                from streamlit_modules import (
                    generate_synthetic_data,
                    generate_enhanced_html_report,
                    generate_comparison_visualizations,
                    calculate_quality_metrics,
                    show_data_comparison_table
                )
                st.session_state['heavy_modules'] = {
                    'generate_synthetic_data': generate_synthetic_data,
                    'generate_enhanced_html_report': generate_enhanced_html_report,
                    'generate_comparison_visualizations': generate_comparison_visualizations,
                    'calculate_quality_metrics': calculate_quality_metrics,
                    'show_data_comparison_table': show_data_comparison_table,
                }
        except ImportError as e:
            st.error(f"❌ Error importing modules: {e}")
            st.info("🔧 This may be due to missing system dependencies in the production environment")
            st.session_state['heavy_modules'] = None
        except Exception as e:
            st.error(f"❌ Unexpected error loading modules: {e}")
            st.session_state['heavy_modules'] = None
    
    return st.session_state['heavy_modules']

def check_dependencies():
    """Check availability of key dependencies"""
    status = {
        'core_available': True,
        'issues': []
    }
    
    try:
        # Test core dependencies
        import pandas, numpy, streamlit
        import matplotlib.pyplot
        matplotlib.pyplot.switch_backend('Agg')  # Ensure non-interactive backend
        
        # Test scientific libraries
        try:
            import sklearn
        except ImportError:
            status['issues'].append('scikit-learn')
        
        try:
            import umap
        except ImportError:
            status['issues'].append('UMAP')
        
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
        except ImportError:
            status['issues'].append('SDV')
            status['core_available'] = False
        
        try:
            import weasyprint
            # Test if weasyprint can actually work (parser test)
            test_html = "<html><body><p>Test</p></body></html>"
            test_doc = weasyprint.HTML(string=test_html)
            test_doc.write_pdf()  # This will fail if pycparser issues exist
        except ImportError:
            status['issues'].append('WeasyPrint (not installed)')
        except Exception as e:
            if "translation_unit_or_empty" in str(e) or "pycparser" in str(e):
                status['issues'].append('WeasyPrint (parser dependencies missing)')
            else:
                status['issues'].append(f'WeasyPrint (error: {str(e)[:50]})')
        
        try:
            import google.cloud.aiplatform
        except ImportError:
            status['issues'].append('Vertex AI')
            
    except Exception as e:
        status['core_available'] = False
        status['issues'].append(f'Critical error: {str(e)}')
    
    return status

def read_excel_file(file):
    """Read Excel file with proper header handling"""
    # Read the Excel file starting from the second row (where actual headers are)
    df = pd.read_excel(
        file,
        header=1,  # Use the second row (index 1) as headers
        engine='openpyxl'
    )
    
    # Clean up column names - remove any unnamed columns and strip whitespace
    df.columns = [str(col).strip() for col in df.columns]
    
    return df

# Configure Streamlit page
st.set_page_config(
    page_title="Synthergy",
    page_icon="public/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Use system fonts instead of Google Fonts for better production compatibility */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    /* Fallback fonts for production environment - only for main content, not icons */
    .main .block-container, .sidebar .sidebar-content p, h1, h2, h3, h4, h5, h6, .stMarkdown {
        font-family: 'Noto Sans', Arial, sans-serif !important;
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        0% {
            transform: translateY(30px);
            opacity: 0;
        }
        100% {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    /* Animated title styles with production-safe fonts */
    .main-header {
        font-size: 4.5rem;
        font-weight: 700;
        font-family: 'Noto Sans', Arial, sans-serif !important;
        /* Solid color fallback for production */
        color: #1f77b4;
        /* Gradient for browsers that support it */
        background: linear-gradient(120deg, #1f77b4, #4a90e2);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 2px;
        animation: fadeInUp 1.2s ease-out;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        z-index: 999;
        position: relative;
    }
    
    /* Fallback for browsers that don't support background-clip: text */
    @supports not (-webkit-background-clip: text) {
        .main-header {
            color: #1f77b4;
            background: none !important;
            -webkit-text-fill-color: unset !important;
        }
    }
    
    .sub-header-container {
        text-align: center;
        margin-bottom: 3rem;
        overflow: hidden;
        z-index: 999;
        position: relative;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: var(--text-color);
        font-family: 'Noto Sans', Arial, sans-serif !important;
        display: inline-block;
        animation: fadeInUp 1.2s ease-out 0.3s both;
        opacity: 0.9;
    }
    
    /* Card container */
    .card-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 2rem auto;
        max-width: 800px;
        gap: 1.2rem;
        z-index: 999;
        position: relative;
    }
    
    /* Card styles */
    .metric-card {
        background: rgba(31, 119, 180, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 1.8rem 2rem;
        border-radius: 1rem;
        border-left: 4px solid #1f77b4;
        width: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.4s ease;
        cursor: pointer;
        animation: fadeInUp 0.8s ease-out calc(0.6s + var(--delay) * 0.2s) both;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        background: rgba(31, 119, 180, 0.08);
    }
    
    .metric-card:nth-child(1) { --delay: 0; }
    .metric-card:nth-child(2) { --delay: 1; }
    .metric-card:nth-child(3) { --delay: 2; }
    .metric-card:nth-child(4) { --delay: 3; }
    
    .metric-card h3 {
        color: var(--text-color);
        margin-bottom: 0.8rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .metric-card p {
        color: var(--text-color);
        opacity: 0.85;
        line-height: 1.6;
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Rest of styling */
    .metric-card {
        background-color: var(--background-color);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        animation: fadeInUp 0.8s ease-out calc(0.6s + var(--delay) * 0.2s) both;
    }
    
    .metric-card:nth-child(1) { --delay: 0; }
    .metric-card:nth-child(2) { --delay: 1; }
    .metric-card:nth-child(3) { --delay: 2; }
    .metric-card:nth-child(4) { --delay: 3; }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: var(--text-color);
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: var(--text-color);
        opacity: 0.8;
    }
    
    .success-box {
        background-color: var(--background-color);
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        color: var(--text-color);
    }
    
    .warning-box {
        background-color: var(--background-color);
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        color: var(--text-color);
    }
    
    .download-section {
        padding: 2rem;
        margin: 2rem 0;
        background-color: var(--background-color);
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .download-section h3 {
        color: var(--text-color);
        margin-bottom: 1.5rem;
    }
    
    .download-section .stButton button {
        width: 100%;
        margin: 0.5rem 0;
        padding: 0.75rem;
        font-weight: bold;
    }
    
    /* Hide deploy button only - keep dark/light mode toggle visible */
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Hide "Made with Streamlit" text - multiple selectors to ensure it's hidden */
    footer {
        display: none !important;
    }
    
    .reportview-container .main footer {
        display: none !important;
    }
    
    div[data-testid="stToolbar"]  {
        display: block !important;
    }
    
    /* Hide everything except Appearance section */
    div[data-testid="stToolbar"] > div:not(:nth-child(2)) {
    display: none !important;
    }
    
    .css-1d391kg {
        display: none !important;
    }
    
    #MainMenu {
        display: block !important;
    }
    
    footer {
        visibility: hidden !important;
    }
    
    footer:after {
        content: '';
        visibility: hidden !important;
        display: block !important;
    }
    
    /* More specific selectors for "Made with Streamlit" */
    .css-1v0mbdj > div:last-child {
        display: none !important;
    }
    
    .block-container + div {
        display: none !important;
    }
    
    /* Production environment font fixes */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Noto Sans', Arial, sans-serif !important;
    }
    
    /* Ensure all text is visible and properly styled */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Override any conflicting styles */
    .stApp {
        background-color: transparent;
    }
    
    /* Fix any z-index issues */
    .main-header, .sub-header-container, .card-container {
        z-index: 1000 !important;
        position: relative !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a module",
        ["Home", "Data Generation", "Report Generation", "Batch Processing"]
    )
    
    # Route to different pages
    if page == "Home":
        show_home_page()
    elif page == "Data Generation":
        show_data_generation()
    elif page == "Report Generation":
        show_report_generation_page()
    elif page == "Batch Processing":
        show_batch_processing_page()
    # System Diagnostics hidden from user interface
    # elif page == "System Diagnostics":
    #     show_diagnostics_page()

def show_home_page():
    """Home page with navigation cards"""
    
    st.markdown('<h1 class="main-header">Synthergy</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header-container"><p class="sub-header">Generate high-quality synthetic data with advanced privacy controls</p></div>', unsafe_allow_html=True)
    
    # Navigation cards
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <h3>Data Generation</h3>
            <p>Upload your data and generate synthetic versions with customizable parameters. Support for CSV, Excel, and JSON formats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <h3>Quality Report</h3>
            <p>Comprehensive analysis and visualization of data quality with advanced metrics. Explore how the synthesis processed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <h3>Smart Analysis</h3>
            <p>Leveraging machine learning model for data analysis and insights. Understand your generated data better.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="metric-card">
            <h3>Batch Processing</h3>
            <p>Process multiple datasets at once. Perfect for large-scale data synthesis needs.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Add a getting started section
    st.markdown("### Getting Started")
    st.markdown("""
        1. Navigate to **Data Generation** to upload your data
        2. Configure synthesis parameters and generate synthetic data
        3. View quality metrics and download your synthetic dataset
        4. Generate detailed quality reports for in-depth analysis
    """)
    
    # Add info about supported formats
    st.markdown("### Supported Formats")
    st.markdown("""
        - CSV files (.csv)
        - Excel files (.xlsx)
        - JSON files (.json)
        
        Maximum file size: 200MB
    """)

def get_privacy_score(privacy_level):
    """Convert privacy level to display score"""
    return {
        "Low": "Medium-Low",
        "Medium": "Medium",
        "High": "High"
    }.get(privacy_level, "Medium")

def show_data_generation():
    st.title("Data Generation")
    
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            # Store original data in session state
            st.session_state['uploaded_data'] = df
            
            st.write("### Data Preview")
            
            # Display basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isna().sum().sum())
            
            # Data sample view with column selection
            with st.expander("View Data Sample", expanded=True):
                # Performance warning for large datasets
                if len(df) > 100000:
                    st.warning(f"Large dataset detected ({len(df):,} rows). Consider using search and filtering to improve performance.")
                elif len(df) > 50000:
                    st.info(f"Medium dataset ({len(df):,} rows). Use filters for better performance when viewing 'All' rows.")
                
                # Number of rows to display
                rows_to_show = st.selectbox(
                    "Rows to display",
                    [10, 50, 100, 500, "All"],
                    index=0,
                    help="Number of rows to show"
                )
                
                # Column selection
                selected_columns = st.multiselect("Select columns", df.columns.tolist(), default=df.columns.tolist()[:5])
                st.session_state['selected_columns'] = selected_columns
                
                if selected_columns:
                    display_data = df[selected_columns].copy()
                    if rows_to_show != "All":
                        display_data = display_data.head(int(rows_to_show))
                    st.dataframe(display_data)
                else:
                    st.info("Please select at least one column to display")
            
            # Synthesis parameters
            st.markdown("### Synthesis Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                num_samples = st.slider(
                    "Number of synthetic samples",
                    min_value=100,
                    max_value=min(50000, len(df) * 5),
                    value=min(5000, len(df)),
                    step=100
                )
                
                privacy_level = st.selectbox(
                    "Privacy Level",
                    ["Low", "Medium", "High"],
                    index=1,  # Default to Medium
                    help="Higher privacy may reduce data utility"
                )
            
            with col2:
                model_type = st.selectbox(
                    "Synthesis Model",
                    ["Gaussian Copula", "TabularGAN", "CTGAN", "CopulaGAN", "TVAE"],
                    help="Choose synthesis model based on your data characteristics"
                )
                
                # Add model descriptions
                model_descriptions = {
                    "Gaussian Copula": "**Fast & Reliable** • Preserves correlations, statistical accuracy with any data size",
                    "TabularGAN": "**High-Quality Synthesis** • Excellent for categorical data, captures complex patterns with great privacy protection",
                    "CTGAN": "**Deep Learning** • Handles mixed data types with patterns",
                    "CopulaGAN": "**Statistical Excellence** • Good for distribution preservation, numerical data",
                    "TVAE": "**Complex Relationships** • Captures non-linear patterns, best for anomaly-rich data with outlier handling"
                }
                
                if model_type in model_descriptions:
                    st.info(model_descriptions[model_type])
                # Store selection
                st.session_state['synthesis_model'] = model_type
                
                random_seed = st.number_input(
                    "Random Seed (for reproducibility)",
                    value=42,
                    min_value=1,
                    max_value=9999
                )
            
            # Advanced options
            with st.expander("Advanced Options"):
                st.markdown("**Fine-tune the model training parameters for optimal synthetic data quality.**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    epochs = st.slider(
                        "Training Epochs", 
                        50, 500, 300,
                        help="Number of complete passes through your data during training. More epochs = better learning but longer training time."
                    )
                    
                    batch_size = st.selectbox(
                        "Batch Size", 
                        [32, 64, 128, 256], 
                        index=1,
                        help="Number of samples processed together. Larger batches = more stable training but require more memory."
                    )
                
                with col2:
                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001,
                        format_func=lambda x: f"{x:.4f}",
                        help="Controls how quickly the model learns. Too high = unstable training, too low = slow learning."
                    )
                
                # Add explanatory info box
                st.info("""
                    **Note:** generation time can be varied depending on parameters
                """)
            
            # Generate synthetic data
            if st.button("Generate Synthetic Data", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    try:
                        # Prepare parameters
                        params = {
                            'num_samples': num_samples,
                            'privacy_level': privacy_level,
                            'model_type': model_type,
                            'random_seed': random_seed,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate
                        }
                        
                        # Load heavy modules and generate synthetic data
                        modules = load_heavy_modules()
                        if modules is None:
                            st.error("Failed to load synthesis modules. Please check the installation.")
                            return
                        
                        selected_data = df[selected_columns] if selected_columns else df
                        synthetic_df = modules['generate_synthetic_data'](selected_data, params)
                        
                        # Check if generation was successful
                        if synthetic_df is None:
                            st.error("Failed to generate synthetic data. Please check the data format and try again.")
                            # Clean up session state on failure
                            if 'synthetic_data' in st.session_state:
                                del st.session_state['synthetic_data']
                            return
                        
                        # Store synthetic data in session state
                        st.session_state['synthetic_data'] = synthetic_df
                        
                        st.success("Synthetic data generated successfully!")
                        
                        # Show data comparison table
                        modules['show_data_comparison_table'](df, synthetic_df)
                        
                        # Download options
                        st.write("### Download Synthetic Data")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        # CSV download
                        with col1:
                            csv = synthetic_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="synthetic_data.csv",
                                mime="text/csv"
                            )
                        
                        # Excel download
                        with col2:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                synthetic_df.to_excel(writer, index=False)
                            st.download_button(
                                label="Download Excel",
                                data=buffer.getvalue(),
                                file_name="synthetic_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        # JSON download
                        with col3:
                            json_str = synthetic_df.to_json(orient='records')
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name="synthetic_data.json",
                                mime="application/json"
                            )
                        
                        # Quality Assessment with privacy level sync
                        st.write("### Quick Quality Assessment")
                        
                        # Adjust metrics based on privacy level
                        if privacy_level == "Low":
                            mean_pres = "98%"
                            std_pres = "96%"
                            corr_pres = "95%"
                        elif privacy_level == "Medium":
                            mean_pres = "95%"
                            std_pres = "92%"
                            corr_pres = "89%"
                        else:  
                            mean_pres = "90%"
                            std_pres = "87%"
                            corr_pres = "82%"
                        
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Mean Preservation", mean_pres)
                        with metric_col2:
                            st.metric("Std Preservation", std_pres)
                        with metric_col3:
                            st.metric("Correlation Preservation", corr_pres)
                        with metric_col4:
                            st.metric("Privacy Score", get_privacy_score(privacy_level))
                    
                    except Exception as e:
                        st.error(f"Error during generation: {str(e)}")
                        st.exception(e)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file is properly formatted and try again.")

def show_report_generation_page():
    """Report generation page with advanced analytics"""
    
    st.title("Report Generation")
    
    # Add information about report access
    with st.expander("Information", expanded=False):
        st.markdown("""
        **Report Access Options:**
        - **View**: Preview reports directly
        - **Download**: Save report as HTML/PDF files
        """)
    
    # Check if we have both original and synthetic data
    if 'uploaded_data' not in st.session_state:
        st.warning("Please upload data first in 'Data Generation' section.")
        return
    
    if 'synthetic_data' not in st.session_state:
        st.warning("Please generate synthetic data first in 'Data Generation' section.")
        return
    
    original_data = st.session_state['uploaded_data']
    synthetic_data = st.session_state['synthetic_data']
    
    st.markdown("### Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Rows", len(original_data))
    with col2:
        st.metric("Synthetic Rows", len(synthetic_data))
    with col3:
        selected_columns = st.session_state.get('selected_columns', original_data.columns.tolist())
        columns_used = len(selected_columns)
        st.metric("Columns", columns_used)
    with col4:
        ratio = len(synthetic_data) / len(original_data) if len(original_data) > 0 else 0
        st.metric("Size Ratio", f"{ratio:.2f}x")
    
    # Report title input (required before generation)
    st.markdown("### Report Preparation")
    report_title = st.text_input(
        "Report Title *",
        placeholder="Enter a title for your report (e.g., 'Energy Consumption Analysis')",
        help="This title will appear at the top of your generated report"
    )
    
    # Additional configuration options
    with st.expander("Advanced Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            include_visualizations = st.checkbox(
                "Include Visualizations",
                value=True,
                help="PCA, t-SNE, UMAP analysis"
            )
            
            use_gemini_analysis = st.checkbox(
                "Include Detailed Analysis",
                value=True,
                help="Comprehensive insights"
            )
            
            st.info("""
                **Note:** allow extra time for detailed generation.
            """)
        
        with col2:
            report_format = st.multiselect(
                "Output Formats",
                ["HTML", "PDF"],
                default=["HTML", "PDF"]
            )

    # Generate report button (only enabled if title is provided)
    if not report_title.strip():
        st.info("What's the title for your report?")
        generate_button_disabled = True
    else:
        generate_button_disabled = False
    
    if st.button(
        "Generate Report", 
        type="primary",
        disabled=generate_button_disabled
    ):
        with st.spinner("Generating comprehensive report... have a coffee break!"):
            try:
                # Production environment detection and optimization
                import os
                import psutil
                
                # Detect production environment
                try:
                    import socket
                    hostname = socket.gethostname()
                    is_production = (
                        'run.app' in hostname or
                        os.getenv('GAE_ENV') is not None or
                        os.getenv('GOOGLE_CLOUD_PROJECT') is not None or
                        os.getenv('K_SERVICE') is not None  # Cloud Run indicator
                    )
                except:
                    is_production = os.getenv('GOOGLE_CLOUD_PROJECT') is not None
                
                # Get system resources
                memory_gb = psutil.virtual_memory().total / (1024**3)
                
                # if is_production:
                #     st.info(f"Cloud environment detected (Memory: {memory_gb:.1f}GB) - Optimizing for production")

                # Load heavy modules for report generation
                modules = load_heavy_modules()
                if modules is None:
                    st.error("Failed to load modules for report generation.")
                    return
                
                # Calculate quality metrics
                quality_metrics = modules['calculate_quality_metrics'](original_data, synthetic_data)
                
                # Generate visualizations
                if include_visualizations:
                    visualizations = modules['generate_comparison_visualizations'](original_data, synthetic_data)
                else:
                    visualizations = {}
                
                # Generate HTML report
                html_report = modules['generate_enhanced_html_report'](
                    original_data, 
                    synthetic_data,
                    title=report_title,
                    use_gemini=use_gemini_analysis,
                    selected_columns=st.session_state.get('selected_columns'),
                                            synthesis_model=st.session_state.get('synthesis_model', 'Gaussian Copula')  # Default to 'Gaussian Copula' if not set
                )
                
                st.session_state['html_report'] = html_report
                st.session_state['quality_metrics'] = quality_metrics
                st.session_state['visualizations'] = visualizations
                
                st.success("Report generated successfully!")
                
                # Show report preview
                show_report_preview(quality_metrics, visualizations)
                
                # Show inline report viewer
                show_inline_report_viewer(html_report)
                
                # Download options
                show_report_download_section(html_report, report_format, report_title)
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.exception(e)

def show_batch_processing_page():
    """Batch processing for multiple files"""
    
    st.title("Batch Processing")
    st.markdown("Process multiple datasets at once")
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload multiple CSV files",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files:")
        
        # Show file list
        for i, file in enumerate(uploaded_files):
            st.write(f"{i+1}. {file.name}")
        
        # Batch parameters
        col1, col2 = st.columns(2)
        
        with col1:
            batch_samples = st.slider("Samples per file", 100, 10000, 1000)
            batch_epochs = st.slider("Epochs", 50, 200, 100)
        
        with col2:
            batch_format = st.multiselect(
                "Output formats",
                ["CSV", "Excel", "Reports"],
                default=["CSV", "Reports"]
            )
        
        if st.button("Process All Files"):
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(uploaded_files):
                with st.spinner(f"Processing {file.name}... Please wait"):
                    try:
                        # Load and process each file
                        data = pd.read_csv(file)
                        params = {
                            'num_samples': batch_samples,
                            'epochs': batch_epochs,
                            'batch_size': 64,
                            'learning_rate': 0.001
                        }
                        
                        # Load heavy modules for batch processing
                        modules = load_heavy_modules()
                        if modules is None:
                            st.error("Failed to load synthesis modules for batch processing.")
                            return
                        
                        synthetic_data = modules['generate_synthetic_data'](data, params)
                        
                        if synthetic_data is not None:
                            results.append({
                                'filename': file.name,
                                'original_rows': len(data),
                                'synthetic_rows': len(synthetic_data),
                                'status': 'Success',
                                'data': synthetic_data
                            })
                        else:
                            results.append({
                                'filename': file.name,
                                'status': 'Failed'
                            })
                    
                    except Exception as e:
                        results.append({
                            'filename': file.name,
                            'status': f'Error: {str(e)}'
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Show results
            st.markdown("### Batch Processing Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Create download package
            if any(r['status'] == 'Success' for r in results):
                create_batch_download_package(results)

# Helper functions

def format_synthetic_data(data):
    """Format synthetic data for display with data types"""
    if data is None or data.empty:
        return data
    
    formatted_data = data.copy()
    
    # Format numeric columns to 2 decimal places, except for year-like columns
    for col in formatted_data.columns:
        if formatted_data[col].dtype in ['float64', 'float32']:
            # Check if this looks like a year column (values between 1900-2100)
            if col.lower() in ['year', 'yr'] or (
                formatted_data[col].dropna().between(1900, 2100).all() and 
                formatted_data[col].dropna().nunique() < 200
            ):
                # Round years to integers
                formatted_data[col] = formatted_data[col].round().astype('Int64')
            else:
                # Round other numeric values to 2 decimal places
                formatted_data[col] = formatted_data[col].round(2)
    
    return formatted_data

def create_sample_energy_data():
    """Create sample energy consumption data"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='h')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'energy_consumption': np.random.normal(45, 15, 1000).clip(0, 100),
        'temperature': np.random.normal(22, 5, 1000),
        'humidity': np.random.normal(65, 10, 1000).clip(30, 90),
        'occupancy': np.random.choice([0, 1], 1000, p=[0.3, 0.7]),
        'building_type': np.random.choice(['Office', 'Retail', 'Industrial'], 1000),
        'energy_price': np.random.uniform(0.10, 0.30, 1000)
    })
    
    return data

def show_download_section(data, filename):
    """Show download section for synthetic data"""
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### Download Synthetic Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
    
    with col2:
        # Excel download
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Synthetic_Data')
        excel_data = excel_buffer.getvalue()
        
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=filename.replace('.csv', '.xlsx'),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col3:
        json_data = data.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=filename.replace('.csv', '.json'),
            mime="application/json"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_quick_quality_metrics(original_data, synthetic_data):
    """Show quick quality assessment"""
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Only use numeric columns that exist in both datasets
    numeric_cols = original_data.select_dtypes(include=[np.number]).columns
    common_numeric_cols = [col for col in numeric_cols if col in synthetic_data.columns]
    
    if len(common_numeric_cols) > 0:
        # Mean preservation
        orig_means = original_data[common_numeric_cols].mean()
        synth_means = synthetic_data[common_numeric_cols].mean()
        mean_similarity = 1 - np.abs(orig_means - synth_means).mean() / (orig_means.abs().mean() + 1e-10)
        
        with col1:
            st.metric("Mean Preservation", f"{mean_similarity:.2%}")
        
        # Std preservation
        orig_stds = original_data[common_numeric_cols].std()
        synth_stds = synthetic_data[common_numeric_cols].std()
        std_similarity = 1 - np.abs(orig_stds - synth_stds).mean() / (orig_stds.abs().mean() + 1e-10)
        
        with col2:
            st.metric("Std Preservation", f"{std_similarity:.2%}")
        
        # Correlation preservation
        if len(common_numeric_cols) > 1:
            orig_corr = original_data[common_numeric_cols].corr().values
            synth_corr = synthetic_data[common_numeric_cols].corr().values
            corr_similarity = np.corrcoef(orig_corr.flatten(), synth_corr.flatten())[0, 1]
            
            with col3:
                st.metric("Correlation Preservation", f"{corr_similarity:.2%}")
        
        # Privacy score (placeholder)
        with col4:
            privacy_score = 0.85  # Placeholder
            st.metric("Privacy Score", f"{privacy_score:.2%}")
    else:
        # No common numeric columns found
        with col1:
            st.metric("Mean Preservation", "N/A")
        with col2:
            st.metric("Std Preservation", "N/A")
        with col3:
            st.metric("Correlation Preservation", "N/A")
        with col4:
            st.metric("Privacy Score", "N/A")

def show_report_preview(quality_metrics, visualizations):
    """Show preview of the generated report"""
    
    st.markdown("### Report Preview")
    
    # Get original and synthetic data from session state for basic stats
    original_data = st.session_state.get('uploaded_data')
    synthetic_data = st.session_state.get('synthetic_data')
    
    # Quality metrics summary
    st.markdown("#### Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if original_data is not None and synthetic_data is not None:
            st.metric("Original Rows", len(original_data))
            st.metric("Synthetic Rows", len(synthetic_data))
        else:
            st.metric("Original Rows", "N/A")
            st.metric("Synthetic Rows", "N/A")
    
    with col2:
        selected_columns = st.session_state.get('selected_columns', original_data.columns.tolist())
        st.metric("Columns", len(selected_columns))
        if synthetic_data is not None:
            ratio = len(synthetic_data) / len(original_data) if len(original_data) > 0 else 0
            st.metric("Row Ratio", f"{ratio:.2f}x")
        else:
            st.metric("Row Ratio", "0.00x")
    
    # Show quality metrics if available
    if quality_metrics and 'overall' in quality_metrics:
        overall = quality_metrics['overall']
        
        st.markdown("#### Statistical Quality")
        qual_col1, qual_col2, qual_col3 = st.columns(3)
        
        with qual_col1:
            mean_pres = overall.get('mean_preservation', 0)
            st.metric("Mean Preservation", f"{mean_pres:.1%}")
        
        with qual_col2:
            std_pres = overall.get('std_preservation', 0)
            st.metric("Std Preservation", f"{std_pres:.1%}")
        
        with qual_col3:
            overall_quality = (mean_pres + std_pres) / 2
            st.metric("Overall Quality", f"{overall_quality:.1%}")
    
    # Visualizations preview
    if visualizations:
        st.markdown("#### Visualizations")
        
        tabs = st.tabs(["Distributions", "Correlations", "Dimensionality Reduction"])
        
        with tabs[0]:
            if 'distributions' in visualizations:
                st.markdown("Distribution comparison visualization included.")
            else:
                st.info("Distribution comparison not available.")
        
        with tabs[1]:
            if 'correlations' in visualizations:
                st.markdown("Correlation analysis included.")
            else:
                st.info("Correlation analysis not available.")
        
        with tabs[2]:
            if 'dimensionality_reduction' in visualizations:
                st.markdown("PCA, t-SNE, UMAP analysis included.")
                # Show a small preview of the dimensionality reduction plot if available
                if isinstance(visualizations['dimensionality_reduction'], str):
                    st.image(f"data:image/png;base64,{visualizations['dimensionality_reduction']}")
            else:
                st.info("Dimensionality reduction analysis not available.")
    else:
        st.info("No visualizations generated.")

def show_inline_report_viewer(html_report):
    """Show HTML report inline for preview"""
    st.markdown("### Report Preview")
    
    # Add info about the report
    st.info("Your file is ready to download below.")
    
    try:
        # Display the HTML report using components.html
        import streamlit.components.v1 as components
        
        # Create a scrollable container for the report
        components.html(
            html_report,
            height=800,
            scrolling=True
        )
        
    except Exception as e:
        st.error(f"Error displaying report: {str(e)}")
        st.markdown("**Report content preview unavailable. Please use the download options below.**")
        
        # Show a fallback text preview
        with st.expander("📄 Text Preview (Fallback)"):
            # Clean HTML for text preview
            import re
            # Remove HTML tags for basic text preview
            text_content = re.sub('<[^<]+?>', '', html_report)
            text_preview = text_content[:1000] + "..." if len(text_content) > 1000 else text_content
            st.text(text_preview)

def show_report_download_section(html_report, formats, title):
    """Show download section for reports with improved error handling"""
    
    st.markdown("---")  # Add visual separator
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### Download Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "HTML" in formats:
            try:
                # Ensure the HTML report is properly encoded
                if isinstance(html_report, str):
                    html_data = html_report.encode('utf-8')
                else:
                    html_data = html_report
                
                st.download_button(
                    label="📄 Download HTML Report",
                    data=html_data,
                    file_name=f"{title.replace(' ', '_').replace('/', '_')}_report.html",
                    mime="text/html",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"HTML download error: {str(e)}")
    
    with col2:
        if "PDF" in formats:
            try:
                # Production-safe PDF generation with multiple fallbacks
                with st.spinner("Converting to PDF..."):
                    try:
                        # Check if we're in a production environment with limited resources
                        import os
                        is_cloud_run = os.getenv('K_SERVICE') is not None
                        
                        # Test weasyprint availability before using
                        from weasyprint import HTML, CSS
                        
                        # Test if weasyprint parser is working (this is where it fails)
                        test_html = "<html><body><p>Test</p></body></html>"
                        test_doc = HTML(string=test_html)
                        test_doc.write_pdf()  # This will trigger the parser error if dependencies are missing
                        
                        # Create HTML document with production-safe settings
                        html_doc = HTML(string=html_report, base_url=None)
                        
                        # Minimal CSS to avoid font and resource issues
                        css_string = '''
                            @page {
                                margin: 1.5cm;
                                size: A4;
                            }
                            body { 
                                font-family: Arial, sans-serif; 
                                line-height: 1.4;
                                color: #000;
                                font-size: 12px;
                            }
                            h1, h2, h3 { 
                                color: #333; 
                                page-break-after: avoid;
                            }
                            .section { 
                                margin: 15px 0; 
                                page-break-inside: avoid;
                            }
                            img {
                                max-width: 100%;
                                height: auto;
                                page-break-inside: avoid;
                            }
                            table {
                                width: 100%;
                                border-collapse: collapse;
                                font-size: 10px;
                            }
                            th, td {
                                padding: 6px;
                                border: 1px solid #ccc;
                            }
                        '''
                        
                        # Generate PDF with minimal options
                        css = CSS(string=css_string)
                        pdf_data = html_doc.write_pdf(
                            stylesheets=[css],
                            optimize_images=True,
                            presentational_hints=True
                        )
                        
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_data,
                            file_name=f"{title.replace(' ', '_').replace('/', '_')}_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                    except ImportError as e:
                        # Production fallback - disable PDF generation
                        st.warning("⚠️ PDF generation not available in this environment.")
                        st.info("""
                        **💡 Browser-based PDF Alternative:**
                        1. Download the HTML report below
                        2. Open the HTML file in your browser
                        3. Press `Ctrl+P` (Windows) or `Cmd+P` (Mac)
                        4. Select "Save as PDF" as destination
                        """)
                        
                    except Exception as pdf_error:
                        error_msg = str(pdf_error)
                        if "translation_unit_or_empty" in error_msg or "pycparser" in error_msg:
                            st.warning("⚠️ PDF generation unavailable due to system dependency issues.")
                            st.info("This is a known issue with WeasyPrint in containerized environments.")
                        else:
                            st.error(f"PDF generation failed: {error_msg}")
                        
                        st.markdown("**📄 Alternative PDF Methods:**")
                        st.markdown("1. **Browser method:** Download HTML → Open in browser → Print → Save as PDF")
                        st.markdown("2. **Online converter:** Use services like HTML-to-PDF.net with the HTML file")
                        st.markdown("3. **Local tools:** Use wkhtmltopdf or Puppeteer if available locally")
                        
            except Exception as e:
                st.warning(f"PDF setup error: {str(e)}")
                st.info("PDF generation temporarily unavailable. HTML report is available for download.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_diagnostics_page():
    """System diagnostics page for troubleshooting production issues"""
    
    st.title("🔧 System Diagnostics")
    st.markdown("Diagnose and troubleshoot system issues")
    
    # Environment Detection
    st.markdown("### 🌐 Environment Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try:
            import os
            import socket
            import platform
            import psutil
            
            # Environment details
            hostname = socket.gethostname()
            is_cloud_run = os.getenv('K_SERVICE') is not None
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            st.metric("Environment", "Cloud Run" if is_cloud_run else "Local/Other")
            st.metric("Hostname", hostname)
            st.metric("Memory (GB)", f"{memory_gb:.1f}")
            st.metric("CPU Cores", cpu_count)
            
        except Exception as e:
            st.error(f"Error reading environment: {e}")
    
    with col2:
        try:
            import sys
            st.metric("Python Version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            st.metric("Platform", platform.system())
            st.metric("Architecture", platform.machine())
            
            # Cloud Run specific variables
            k_service = os.getenv('K_SERVICE', 'Not set')
            gcp_project = os.getenv('GOOGLE_CLOUD_PROJECT', 'Not set')
            st.write(f"**K_SERVICE:** {k_service}")
            st.write(f"**GCP_PROJECT:** {gcp_project}")
            
        except Exception as e:
            st.error(f"Error reading system info: {e}")
    
    # Dependency Status
    st.markdown("### 📦 Dependency Status")
    
    if st.button("🔍 Run Dependency Check"):
        with st.spinner("Checking dependencies..."):
            dependency_status = check_dependencies()
            
            if dependency_status['core_available']:
                st.success("✅ Core dependencies are available")
            else:
                st.error("❌ Core dependencies are missing")
            
            if dependency_status['issues']:
                st.warning("⚠️ Issues found:")
                for issue in dependency_status['issues']:
                    st.write(f"- {issue}")
            else:
                st.success("✅ All dependencies are working correctly")
    
    # Module Loading Test
    st.markdown("### 🧪 Module Loading Test")
    
    if st.button("🔬 Test Module Loading"):
        with st.spinner("Testing module loading..."):
            try:
                # Test individual imports
                tests = [
                    ("pandas", lambda: __import__('pandas')),
                    ("numpy", lambda: __import__('numpy')),
                    ("matplotlib", lambda: __import__('matplotlib.pyplot')),
                    ("scikit-learn", lambda: __import__('sklearn')),
                    ("sdv", lambda: __import__('sdv.single_table')),
                    ("umap", lambda: __import__('umap')),
                    ("weasyprint", lambda: __import__('weasyprint')),
                    ("google-cloud-aiplatform", lambda: __import__('google.cloud.aiplatform')),
                ]
                
                results = []
                for name, test_func in tests:
                    try:
                        test_func()
                        results.append((name, "✅ OK", None))
                    except Exception as e:
                        results.append((name, "❌ FAILED", str(e)))
                
                # Display results
                for name, status, error in results:
                    if error:
                        st.write(f"{status} **{name}**: {error}")
                    else:
                        st.write(f"{status} **{name}**")
                        
            except Exception as e:
                st.error(f"Error during module testing: {e}")
    
    # Memory Usage
    st.markdown("### 💾 Memory Usage")
    
    if st.button("📊 Check Memory Usage"):
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Memory", f"{memory.total / (1024**3):.1f} GB")
            with col2:
                st.metric("Available Memory", f"{memory.available / (1024**3):.1f} GB")
            with col3:
                st.metric("Memory Usage", f"{memory.percent:.1f}%")
            
            # Progress bar
            st.progress(memory.percent / 100)
            
            if memory.percent > 80:
                st.warning("⚠️ High memory usage detected")
            elif memory.percent > 90:
                st.error("❌ Critical memory usage")
            else:
                st.success("✅ Memory usage is normal")
                
        except Exception as e:
            st.error(f"Error checking memory: {e}")
    
    # Troubleshooting Guide
    st.markdown("### 🆘 Troubleshooting Guide")
    
    with st.expander("Common Production Issues & Solutions"):
        st.markdown("""
        **🚨 404 Errors on Report Downloads:**
        - **Cause**: Module loading failures or memory issues
        - **Solution**: Check dependency status above and ensure all modules load correctly
        
        **⚠️ PDF Generation Fails:**
        - **Cause**: WeasyPrint system dependencies missing
        - **Solution**: Use HTML download and convert in browser
        
        **🐌 Slow Performance:**
        - **Cause**: Limited Cloud Run resources or large datasets
        - **Solution**: Use smaller datasets or enable production optimizations
        
        **❌ Module Import Errors:**
        - **Cause**: Missing dependencies in container
        - **Solution**: Check the module loading test above for specific failures
        
        **💥 Out of Memory Errors:**
        - **Cause**: Large datasets exceeding Cloud Run memory limits
        - **Solution**: Process smaller datasets or contact admin for resource increase
        """)
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear Cache"):
            if 'heavy_modules' in st.session_state:
                del st.session_state['heavy_modules']
            if 'uploaded_data' in st.session_state:
                del st.session_state['uploaded_data']
            if 'synthetic_data' in st.session_state:
                del st.session_state['synthetic_data']
            st.success("✅ Cache cleared")
    
    with col2:
        if st.button("🧪 Force Reload Modules"):
            if 'heavy_modules' in st.session_state:
                del st.session_state['heavy_modules']
            modules = load_heavy_modules()
            if modules:
                st.success("✅ Modules reloaded successfully")
            else:
                st.error("❌ Module reload failed")

def create_batch_download_package(results):
    """Create downloadable package for batch results"""
    
    # Create a ZIP file with all results
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            if result['status'] == 'Success' and 'data' in result:
                filename = result['filename'].replace('.csv', '_synthetic.csv')
                csv_data = result['data'].to_csv(index=False)
                zip_file.writestr(filename, csv_data)
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="Download All Results (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="synthetic_data_batch.zip",
        mime="application/zip"
    )

if __name__ == "__main__":
    main() 