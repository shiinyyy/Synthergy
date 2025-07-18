# Synthergy - Synthetic Data Platform

A Streamlit-based platform for generating high-quality synthetic datasets using advanced statistical methods and machine learning techniques.

## Features

### **Data Generation**
- **Multi-format Support**: CSV, Excel file uploads
- **Smart Data Processing**: Automatic data type detection and handling
- **Synthesis Options**: Synthetic Data Vault SDK integration with statistical fallback
- **Quality Controls**: Privacy levels, sample size configuration
- **Download Formats**: CSV, Excel, JSON export

### **Report Generation** 
- **Comprehensive Analytics**: Statistical quality metrics
- **Visual Analysis**: PCA, t-SNE, UMAP dimensionality reduction
- **Report**: Clean mono HTML reports
- **Export Options**: HTML and PDF generation
- **Quality Assessment**: Mean/std preservation, correlation analysis

### **Key Capabilities**
- Preserves data types (strings stay strings, years as integers)
- Statistical synthesis fallback when SDV unavailable
- Quality metrics
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
   pip install -r requirements_minimal.txt
   ```

3. **Run Application**
   ```bash
   streamlit run app.py
   ```

4. **Access Interface**
   - Navigate using sidebar: Data Generation → Report Generation

### Testing Steps

1. **Upload Data**: Use CSV/Excel file (or built-in sample data)
2. **Configure Parameters**: Set sample count, privacy level
3. **Generate Synthetic Data**: Click "Generate Synthetic Data"
4. **Review Quality**: Check preservation metrics
5. **Create Report**: Navigate to "Report Generation" → Enter title → "Generate Report"
6. **Download Results**: Export data (CSV/Excel/JSON) and reports (HTML/PDF)

## Dependencies

Core requirements for local testing:
- **streamlit**: Web interface
- **pandas/numpy**: Data processing
- **scikit-learn**: Machine learning algorithms
- **matplotlib/plotly**: Visualizations
- **umap-learn**: Dimensionality reduction

Optional:
- **weasyprint**: PDF generation
- **sdv**: Synthetic Data Vault for advanced synthesis (fallback included)

## Challenges and Solutions

### **Production Visualization Issues**
**Challenge**: Visualizations showed "No valid visualization data available" in production despite working locally.
**Solution**: 
- Implemented 5-fallback display system (temp files, base64, binary data, emergency conversion)
- Fixed HTML report generation to embed base64 images instead of placeholders
- Enhanced matplotlib configuration for container environments

### **scikit-learn Compatibility**
**Challenge**: `TypeError: TSNE.__init__() got an unexpected keyword argument 'n_iter'`
**Solution**: Updated t-SNE parameters from deprecated `n_iter` to `max_iter` for current scikit-learn versions

### **AWS S3 Integration**
**Challenge**: No existing S3 integration despite user expectations for cloud storage.
**Solution**: 
- Implemented complete S3 workflow with input/output bucket separation
- Added automatic file uploads with local fallback capability
- Created timestamped report storage system

### **Container Memory Management** 
**Challenge**: Large visualizations (4MB+) causing memory issues in Cloud Run containers.
**Solution**:
- Optimized image generation and storage formats
- Implemented multiple visualization storage methods
- Added proper memory cleanup and garbage collection

### **Analysis Quality**
**Challenge**: Bedrock analysis was incomplete with poor formatting and missing final assessments.
**Solution**:
- Increased token limits from 4000 to 6000 for comprehensive analysis
- Enhanced prompts with structured 6-section format including "Final Assessment"
- Added explicit HTML formatting requirements for proper list rendering

### **Production UI Cleanliness**
**Challenge**: Technical messages cluttered the production interface.
**Solution**: Removed verbose AWS integration status, S3 upload confirmations, and technical metadata while preserving all functionality

## Architecture

**Hybrid Cloud Architecture:**
- **Google Cloud Run**: Application hosting (4GB RAM, 2 CPU)
- **AWS S3**: File storage with input (`synthergy-in`) and output (`synthergy-reports`) buckets  
- **AWS Bedrock**: AI-powered analysis using Claude 3.5 Sonnet

**File Flow:**
```
Uploads: User → App → s3://synthergy-in/uploads/
Synthetic Data: App → s3://synthergy-reports/synthetic-data/
Reports: App → s3://synthergy-reports/reports/
```

## Notes

- Application includes statistical synthesis fallback if SDV unavailable
- Supports large, batch datasets  
- Generated reports maintain clean formatting suitable for stakeholders
- All core processing happens in Google Cloud with AWS integrations for storage and analysis 