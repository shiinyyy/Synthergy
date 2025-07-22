# Synthergy - Synthetic Data Platform
**Test at**: synthergy.minhducdo.com

A Streamlit-based platform for generating high-quality synthetic datasets using advanced statistical methods and machine learning algorithms.

## Features

### **Data Generation**
- **Multi-format Support**: CSV, Excel file uploads
- **Smart Data Processing**: Automatic data type detection and handling
- **Synthesis Options**: SDV integration with multiple models (Gaussian Copula, CTGAN, TabularGAN, etc.)
- **Quality Controls**: Privacy levels, sample size configuration, model selection
- **Download Formats**: CSV, Excel, JSON export

### **Report Generation** 
- **Comprehensive Analytics**: Statistical quality metrics
- **Visual Analysis**: PCA, t-SNE, UMAP dimensionality reduction
- **Smart Analysis**: Gemini-2.5-pro
- **Report**: Clean HTML
- **Export Options**: HTML and PDF generation
- **Quality Assessment**: Mean/std preservation, correlation analysis

### **Key Capabilities**
- Preserves data types (strings stay strings, years as integers)
- Statistical synthesis fallback when SDV unavailable
- Real-time quality metrics
- Professional report generation for stakeholders

## Quick Start

### Prerequisites
- Python 3.10+ 
- pip package manager

### Installation & Testing

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd synthergy
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Interface**
   - Navigate using sidebar: Data Generation → Report Generation

### Testing Steps

1. **Upload Data**: Use CSV/Excel file (or built-in sample data)
2. **Configure Parameters**: Set sample count, privacy level
3. **Generate Synthetic Data**
4. **Review Quality**: Check preservation metrics
5. **Create Report**: Navigate to "Report Generation" → Enter title → "Generate Report"
6. **Download Results**: Export data (CSV/Excel/JSON) and reports (HTML/PDF)

## Dependencies

Core requirements:
- **streamlit**: Web interface
- **pandas**: Data processing and manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **plotly**: Interactive visualizations
- **umap-learn**: Dimensionality reduction

Synthetic Data Generation:
- **sdv**: Synthetic Data Vault for advanced synthesis
- **scipy**: Scientific computing

Cloud AI Integration:
- **google-cloud-aiplatform**: Vertex AI
- **google-cloud-core**: Google Cloud SDK

Report Generation:
- **weasyprint**: PDF generation (currently in maintenance, use html instead)
- **pycparser**: C parser for WeasyPrint dependencies
- **Pillow>**: Image processing

Performance & Optimization:
- **numba**: JIT compilation for numerical functions
- **joblib**: Parallel processing

## Advanced Features

### **Multiple Synthesis Models**
- **Gaussian Copula**: Fast and reliable for preserving correlations
- **CTGAN**: Deep learning approach for mixed data types
- **TabularGAN**: High-quality synthesis with excellent categorical handling
- **CopulaGAN**: Statistical excellence for distribution preservation
- **TVAE**: Variational autoencoder for complex relationship capture

### **Analysis**
- **Vertex AI Integration**: Detailed quality assessment
- **Insights**: Automated analysis of data quality and privacy trade-offs
- **Model Recommendations**: Suggested optimal synthesis approaches
