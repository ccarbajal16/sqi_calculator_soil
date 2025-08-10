# Installation Guide

This guide provides step-by-step instructions for installing and running the Soil Quality Index (SQI) Calculator with True Ordinary Kriging capabilities.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd sqi_stream
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv sqi_env
   
   # On Windows:
   sqi_env\Scripts\activate
   
   # On macOS/Linux:
   source sqi_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run sqi_app.py
   ```

### Method 2: Using conda

1. **Create conda environment**
   ```bash
   conda create -n sqi_env python=3.9
   conda activate sqi_env
   ```

2. **Install dependencies**
   ```bash
   # Install most packages from conda-forge
   conda install -c conda-forge streamlit pandas numpy scipy scikit-learn geopandas plotly folium rasterio shapely openpyxl matplotlib pillow
   
   # Install remaining packages with pip
   pip install streamlit-folium pykrige xlrd
   ```

3. **Run the application**
   ```bash
   streamlit run sqi_app.py
   ```

## Key Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas & numpy**: Data manipulation and numerical computing
- **scipy**: Scientific computing functions

### Geostatistical Analysis (NEW)
- **pykrige**: True Ordinary Kriging implementation with variance estimation
- **scikit-learn**: Machine learning tools (PCA analysis)

### Geospatial Processing
- **geopandas**: Geospatial data handling
- **shapely**: Geometric operations
- **rasterio**: Raster data I/O (GeoTIFF export)
- **pyproj & fiona**: Coordinate transformations and file I/O

### Visualization
- **plotly**: Interactive charts and plots
- **folium**: Interactive mapping
- **streamlit-folium**: Folium integration for Streamlit
- **matplotlib**: Additional plotting capabilities

### File Support
- **openpyxl & xlrd**: Excel file reading
- **pillow**: Image processing

## Troubleshooting

### Common Installation Issues

1. **PyKrige Installation Fails**
   ```bash
   # Try installing with conda first
   conda install -c conda-forge pykrige
   
   # Or install build dependencies
   pip install numpy scipy
   pip install pykrige
   ```

2. **Geopandas Installation Issues**
   ```bash
   # Use conda-forge channel
   conda install -c conda-forge geopandas
   
   # Or install GDAL first
   pip install GDAL
   pip install geopandas
   ```

3. **Rasterio Installation Problems**
   ```bash
   # Install GDAL system dependencies first (Ubuntu/Debian)
   sudo apt-get install gdal-bin libgdal-dev
   
   # Then install rasterio
   pip install rasterio
   ```

4. **Memory Issues with Large Datasets**
   - Reduce grid resolution in spatial assessment
   - Use smaller study areas
   - Consider using IDW for very large datasets

### Platform-Specific Notes

#### Windows
- Install Microsoft Visual C++ Build Tools if compilation errors occur
- Consider using Anaconda distribution for easier dependency management

#### macOS
- Install Xcode command line tools: `xcode-select --install`
- Use Homebrew for system dependencies if needed

#### Linux
- Install system dependencies for geospatial libraries:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev
  
  # CentOS/RHEL
  sudo yum install gdal-devel proj-devel geos-devel
  ```

## Verification

After installation, verify everything works by running the test script:

```bash
python test_kriging.py
```

This will test the True Ordinary Kriging functionality with both synthetic and real data.

## Performance Optimization

### For Better Performance
1. **Use vectorized backend** (default in True Kriging)
2. **Limit grid resolution** for large areas
3. **Reduce number of sample points** if processing is slow
4. **Use appropriate variogram models** (spherical is usually fastest)

### Memory Usage
- Typical memory usage: 100-500 MB for standard datasets
- Large grids (>100x100) may require 1-2 GB RAM
- Consider reducing grid resolution for memory-constrained systems

## Getting Help

If you encounter issues:

1. Check the error messages in the Streamlit interface
2. Review the console output for detailed error information
3. Ensure all dependencies are properly installed
4. Try running the test script to isolate issues
5. Check that your data format matches the requirements

## Version Compatibility

This application has been tested with:
- Python 3.8, 3.9, 3.10, 3.11
- Streamlit 1.28+
- PyKrige 1.7+
- GeoPandas 0.12+

For best results, use the versions specified in requirements.txt.
