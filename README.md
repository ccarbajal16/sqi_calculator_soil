# Soil Quality Index Calculator

A comprehensive Streamlit application for calculating and mapping soil quality indices using Principal Component Analysis (PCA) and advanced geostatistical interpolation methods, including **True Ordinary Kriging with statistical error modeling**.

## Features

### Core Functionality

- **Data Input**: Upload CSV or Excel files with soil property data
- **Exploratory Analysis**: Interactive correlation analysis and variable selection
- **PCA Analysis**: Automatic weight calculation using Principal Component Analysis
- **SQI Calculation**: Three normalization methods (more_is_better, minus_is_better, range_optimo)
- **Advanced Spatial Assessment**: Three interpolation methods with statistical rigor
- **Results Visualization**: Interactive mapping with interpolated surface overlays
- **Surface Visualization**: Heatmap, contour, and variance overlays
- **Export Capabilities**: Download results as CSV and interpolated rasters as GeoTIFF

### Enhanced Coordinate System Support

- **Multiple CRS Support**: Comprehensive list of coordinate reference systems including:
  - WGS84 Geographic (EPSG:4326)
  - UTM zones (Northern and Southern Hemisphere)
  - Web Mercator (EPSG:3857)
  - NAD83 UTM zones
  - SIRGAS 2000 UTM zones
  - Custom EPSG codes
- **Automatic CRS Suggestions**: Smart detection of likely coordinate systems based on data values
- **Coordinate Validation**: Verification that coordinates are reasonable for selected CRS
- **Seamless Transformation**: All spatial data automatically transformed to WGS84 for web mapping
- **Study Area Support**: Upload polygon shapefiles with automatic CRS handling

### Surface Visualization

- **Interactive Heatmaps**: Continuous surface representation using color gradients
- **Contour Visualization**: Point-based contour representation with quality levels
- **Customizable Overlays**: Toggle between different visualization modes
- **Color-Coded Legend**: Clear interpretation of SQI quality levels
- **Opacity Control**: Adjustable transparency for overlay visualization
- **Layer Management**: Interactive layer control for complex visualizations

## Data Requirements

### Mandatory Columns

- **X, Y**: Spatial coordinates in any supported coordinate system
- **Soil Properties**: At least one numeric soil property column

### Recommended Soil Properties (You may add more)

- pH: Soil acidity/alkalinity
- OM: Organic Matter content
- Clay, Silt, Sand: Soil texture components
- CEC: Cation Exchange Capacity
- BD: Bulk Density
- EC: Electrical Conductivity
- P: Available Phosphorus
- K: Available Potassium

### Optional Study Area

Multiple geospatial formats supported for study area boundary definition:

- **Zipped Shapefile**: Upload ZIP file containing .shp, .shx, .dbf, and .prj files
- **GeoPackage**: Upload .gpkg file (modern single-file format with embedded CRS)
- **GeoJSON**: Upload .geojson file (lightweight web-friendly format, typically WGS84)
- **CRS Compatibility**: Study area polygons automatically transformed to match data CRS
- **Multi-polygon Support**: All formats support multiple polygons within a single file

## Supported File Formats

### Data Files

- CSV files (with customizable separators)
- Excel files (.xlsx, .xls)

### Study Area Files

- **Zipped Shapefiles**: ZIP files containing .shp, .shx, .dbf, and .prj files
- **GeoPackage**: .gpkg files with embedded CRS and attribute information
- **GeoJSON**: .geojson files (typically in WGS84 coordinate system)
- **Automatic CRS detection and transformation** for all formats
- **Multi-polygon support** for complex study areas

## Coordinate Reference Systems

The application supports a wide range of coordinate systems and automatically handles transformations:

### Geographic Coordinates

- WGS84 (EPSG:4326): Latitude/Longitude in decimal degrees

### Projected Coordinates

- **UTM Zones**: Universal Transverse Mercator (Northern and Southern Hemisphere)
- **Web Mercator**: EPSG:3857 (commonly used in web mapping)
- **Regional Systems**: NAD83, SIRGAS 2000, and other regional coordinate systems

### CRS Selection Tips

1. **Geographic Data**: If your coordinates are in decimal degrees (e.g., -75.123, -10.456), select WGS84 Geographic
2. **UTM Data**: If your coordinates are large numbers (e.g., 500000, 8800000), select the appropriate UTM zone
3. **Unknown CRS**: Use the automatic CRS suggestions based on your coordinate values
4. **Custom Systems**: Enter any EPSG code for specialized coordinate systems

## Usage Workflow

1. **Input Data**: Upload your soil data file and select the appropriate coordinate system
2. **Exploratory Analysis**: Review correlations and remove highly correlated variables
3. **PCA Analysis**: Run Principal Component Analysis to determine variable weights
4. **SQI Calculation**: Configure normalization parameters and calculate soil quality indices
5. **Spatial Assessment**: Optionally upload study area polygon and run spatial interpolation
6. **Results**: View interactive maps and download results

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

#### Method 1: Clone from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/ccarbajal16/sqi_calculator.git
cd sqi_calculator

# Create virtual environment (recommended)
python -m venv sqi_env

# Activate virtual environment
# On Windows:
sqi_env\Scripts\activate
# On macOS/Linux:
source sqi_env/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Run the application
streamlit run sqi_app.py
```

#### Method 2: Using conda

```bash
# Create conda environment
conda create -n sqi_env python=3.9
conda activate sqi_env

# Install most packages from conda-forge
conda install -c conda-forge streamlit pandas numpy scipy scikit-learn geopandas plotly folium rasterio shapely openpyxl matplotlib pillow

# Install remaining packages with pip
pip install streamlit-folium pykrige xlrd

# Run the application
streamlit run sqi_app.py
```

### Verify Installation

```bash
# Test all dependencies and kriging functionality
python validate_requirements.py
```

### Access the Application

Once running, open your web browser and navigate to:

- **Local URL**: http://localhost:8501
- **Network URL**: http://your-ip:8501 (for network access)

## Technical Details

### Coordinate Transformation

- Uses GeoPandas and PyProj for accurate coordinate transformations
- All spatial data transformed to WGS84 (EPSG:4326) for web mapping display
- Maintains original CRS information for analysis and export

### Interpolation Methods

#### 1. Inverse Distance Weighting (IDW)

- Simple distance-based interpolation
- Fast processing for quick visualization
- Configurable power parameter

#### 2. Ordinary Kriging (Simplified)

- Cubic interpolation using scipy.griddata
- Fast processing with smooth results
- Good for visual presentation

#### 3. True Ordinary Kriging

- **Full geostatistical implementation** using PyKrige library
- **Automatic variogram modeling** with 5 model options
- **Statistical error estimation** (kriging variance)
- **Uncertainty quantification** for scientific applications
- **Professional-grade output** suitable for research and environmental assessments

### Export Formats

- **CSV**: Tabular results with calculated SQI values
- **GeoTIFF**: Georeferenced raster files for GIS analysis

## 📖 Usage Guide

### Step-by-Step Workflow

1. **Data Input** 📁

   - Upload CSV or Excel files with soil property data
   - Ensure X, Y coordinates and soil properties are included
   - Optional: Upload study area polygon (Shapefile, GeoPackage, or GeoJSON)
2. **Exploratory Analysis** 🔍

   - Review data statistics and distributions
   - Identify and remove highly correlated variables (>0.98)
   - Examine spatial distribution of sample points
3. **PCA Analysis** 📊

   - Perform Principal Component Analysis
   - Review component loadings and explained variance
   - Use PCA weights for objective SQI calculation
4. **SQI Calculation** ⚖️

   - Configure normalization parameters for each soil property
   - Choose appropriate normalization method:
     - `more_is_better`: Higher values = better quality (e.g., Organic Matter)
     - `minus_is_better`: Lower values = better quality (e.g., Bulk Density)
     - `range_optimo`: Optimal range (e.g., pH 6.5-7.5)
5. **Spatial Assessment** 🗺️

   - **Choose interpolation method**:
     - **IDW**: Quick visualization, configurable power parameter
     - **Simplified Kriging**: Smooth surfaces, fast processing
     - **True Kriging**: Statistical rigor, uncertainty quantification
   - Configure grid resolution and parameters
   - Apply study area clipping if polygon provided
6. **Results & Export** 📈

   - Interactive maps with multiple overlay options
   - Export SQI data as CSV
   - Export interpolated surfaces as GeoTIFF
   - **NEW**: Export kriging variance for uncertainty analysis

### When to Use Each Interpolation Method

#### Use IDW when:

- Quick visualization is needed
- Simple distance-based interpolation is sufficient
- Processing speed is critical

#### Use Simplified Kriging when:

- Smooth, visually appealing surfaces are desired
- Fast processing with good visual results
- Presentation quality is more important than statistical rigor

#### Use True Ordinary Kriging when:

- **Scientific accuracy is critical**
- **Uncertainty quantification is needed**
- Publishing research or environmental assessments
- Regulatory compliance requires statistical validation
- Risk analysis depends on prediction confidence

## 🧪 Testing & Validation

### Validate Your Installation

```bash
# Test all dependencies and kriging functionality
python validate_requirements.py
```

### 📚 Documentation (docs)

- **USER_GUIDE.md**: Comprehensive user manual with detailed explanations
- **INSTALLATION.md**: Detailed installation guide with troubleshooting
- **TRUE_KRIGING_IMPLEMENTATION.md**: Technical documentation of kriging features

## Troubleshooting

### Installation Issues

#### PyKrige Installation Fails

```bash
# Try conda first
conda install -c conda-forge pykrige

# Or install build dependencies
pip install numpy scipy
pip install pykrige
```

#### Geospatial Libraries Issues

```bash
# Use conda-forge for better compatibility
conda install -c conda-forge geopandas rasterio shapely pyproj fiona

# Linux: Install system dependencies
sudo apt-get install gdal-bin libgdal-dev libproj-dev libgeos-dev
```

### Application Issues

1. **Incorrect Map Display**: Verify CRS matches your data coordinates
2. **Missing Study Area**: Ensure shapefile is properly zipped with all components
3. **Kriging Errors**: Check data quality and reduce grid resolution if needed
4. **Memory Issues**: Use smaller grids or fewer sample points for large datasets

### Getting Help

- Run `python validate_requirements.py` to diagnose installation issues
- Check console output for detailed error messages
- Review the comprehensive documentation files
- Ensure data format matches requirements

## 🤝 Contributing

We welcome contributions to improve the SQI Calculator! Here's how you can help:

### Types of Contributions

- **Bug Reports**: Report issues with detailed error messages
- **Feature Requests**: Suggest new interpolation methods or analysis tools
- **Code Contributions**: Submit pull requests with improvements
- **Documentation**: Help improve guides and examples
- **Testing**: Test with different datasets and report results

### Development Setup

```bash
# Fork the repository on GitHub
git clone https://github.com/yourusername/sqi_stream.git
cd sqi_stream

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .

# Run tests
python validate_requirements.py
python test_kriging.py
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyKrige**: For providing the excellent kriging implementation
- **Streamlit**: For the amazing web application framework
- **GeoPandas**: For powerful geospatial data handling
- **Plotly**: For interactive visualization capabilities

## 📞 Contact & Support

- **Created by Carlos Carbajal, with heartfelt appreciation for the soil science community—whose work profoundly grounds and sustains us.**
- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/ccarbajal16/sqi_calculator/issues)
- **Documentation** : Detailed guides can be found in the repository’s `docs` folder.

---

*Turn your soil data into practical, decision-ready insights using expert-level geostatistical analysis.*

## 📦 Dependencies

### Core Libraries

- **streamlit>=1.28.0** - Web application framework
- **pandas>=1.5.0** - Data manipulation
- **numpy>=1.20.0** - Numerical computing
- **scipy>=1.9.0** - Scientific computing

### Geostatistical Analysis 

- **pykrige>=1.7.0** - True Ordinary Kriging implementation

### Geospatial Processing

- **geopandas>=0.12.0** - Geospatial data handling
- **shapely>=1.8.0** - Geometric operations
- **rasterio>=1.3.0** - Raster data I/O
- **pyproj>=3.4.0** - Coordinate transformations
- **fiona>=1.8.0** - Geospatial file I/O

### Visualization

- **plotly>=5.10.0** - Interactive charts
- **matplotlib>=3.5.0** - Additional plotting
- **folium>=0.14.0** - Interactive mapping
- **streamlit-folium>=0.11.0** - Folium integration

### File Support

- **openpyxl>=3.0.0** - Excel files (new format)
- **xlrd>=2.0.0** - Excel files (legacy format)
- **pillow>=9.0.0** - Image processing

### Machine Learning

- **scikit-learn>=1.1.0** - PCA analysis
