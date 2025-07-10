# Synthergy - Synthetic Data Platform

A Streamlit-based platform for generating high-quality synthetic datasets using advanced statistical methods and machine learning techniques.

## Features

### **Data Generation**
- **Multi-format Support**: CSV, Excel file uploads
- **Smart Data Processing**: Automatic data type detection and handling
- **Synthesis Options**: YData SDK integration with statistical fallback
- **Quality Controls**: Privacy levels, sample size configuration
- **Download Formats**: CSV, Excel, JSON export

### **Report Generation** 
- **Comprehensive Analytics**: Statistical quality metrics
- **Visual Analysis**: PCA, t-SNE, UMAP dimensionality reduction
- **Professional Reports**: Clean black/white HTML reports
- **Export Options**: HTML and PDF generation
- **Quality Assessment**: Mean/std preservation, correlation analysis

### **Key Capabilities**
- Preserves data types (strings stay strings, years as integers)
- Statistical synthesis fallback when YData unavailable
- Real-time quality metrics (95%+ preservation rates)
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
4. **Review Quality**: Check preservation metrics (95%, 92%, 89%)
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
- **ydata-sdk**: Professional synthesis (fallback included)

## Notes

- Application includes statistical synthesis fallback if YData SDK unavailable
- Supports energy consumption, financial, and general tabular datasets
- Generated reports maintain professional formatting suitable for stakeholders
- All data processing happens locally (no external API calls required) 