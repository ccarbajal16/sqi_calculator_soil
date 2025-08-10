# User Guide: Soil Quality Index (SQI) Calculator

This guide provides a comprehensive walkthrough of the Streamlit-based Soil Quality Index (SQI) Calculator. It covers everything from data input and analysis to result visualization and export.

## Table of Contents

- 1. Overview
- 2. Installation & System Requirements
- 3. Data Requirements
  - 3.1 Soil Data File
  - 3.2 Study Area File (Optional)
  - 3.3 Coordinate Reference Systems (CRS)
- 4. Step-by-Step Workflow
  - Step 1: Input Data
  - Step 2: Exploratory Data Analysis
  - Step 3: PCA Analysis
  - Step 4: SQI Calculation
  - Step 5: Spatial Assessment
  - Step 6: Results
- 5. Key Functions Explained
- 6. Core Libraries Used
- 7. Troubleshooting

---

## 1. Overview

The Soil Quality Index (SQI) Calculator is a web application designed to streamline the process of assessing soil health. It uses a data-driven approach, leveraging Principal Component Analysis (PCA) to derive objective weights for various soil properties. The application guides the user through a 6-step workflow, from initial data upload to the final visualization and export of spatial soil quality maps.

### Key Features:
- **Guided Workflow**: A multi-page interface that walks you through each stage of the analysis.
- **Flexible Data Input**: Supports CSV and Excel files for soil data, and multiple geospatial formats (Shapefile, GeoPackage, GeoJSON) for study area boundaries.
- **Advanced CRS Handling**: Includes a comprehensive list of coordinate systems, with automatic suggestions and validation to ensure spatial accuracy.
- **Data-Driven Weighting**: Employs PCA to calculate weights for soil indicators, removing subjectivity from the analysis.
- **Multiple Normalization Methods**: Offers three distinct methods (`more_is_better`, `minus_is_better`, `range_optimo`) to score soil properties based on agronomic principles.
- **Geostatistical Interpolation**: Provides Inverse Distance Weighting (IDW) and a simplified Ordinary Kriging to create continuous soil quality surfaces.
- **Interactive Visualization**: Generates interactive maps with heatmap and contour overlays, allowing for detailed exploration of results.
- **Robust Export Options**: Allows users to download the final dataset with SQI scores (CSV) and the interpolated map as a georeferenced raster (GeoTIFF) for use in GIS software.

## 2. Installation & System Requirements

To run the application locally, you need Python and the required libraries installed.

1.  **Clone the repository or download the source code.**
2.  **Install dependencies** from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application** using Streamlit:
    ```bash
    streamlit run sqi_app.py
    ```

The application will open in a new tab in your web browser.

## 3. Data Requirements

### 3.1 Soil Data File

-   **Format**: CSV (`.csv`) or Excel (`.xlsx`, `.xls`).
-   **Mandatory Columns**:
    -   `X`: The horizontal spatial coordinate (e.g., Longitude, Easting).
    -   `Y`: The vertical spatial coordinate (e.g., Latitude, Northing).
-   **Soil Property Columns**: At least one, but preferably multiple, numeric columns representing soil indicators. Recommended properties include:
    -   `pH`: Soil acidity
    -   `OM`: Organic Matter
    -   `CEC`: Cation Exchange Capacity
    -   `BD`: Bulk Density
    -   `EC`: Electrical Conductivity
    -   `P`: Available Phosphorus
    -   `K`: Available Potassium
    -   `Clay`, `Silt`, `Sand`: Soil texture components

### 3.2 Study Area File (Optional)

You can provide a polygon file to define the boundary for spatial interpolation. This ensures the final map is clipped to your specific area of interest.

-   **Supported Formats**:
    -   **Zipped Shapefile (`.zip`)**: Must contain `.shp`, `.shx`, `.dbf`, and ideally a `.prj` file for CRS information.
    -   **GeoPackage (`.gpkg`)**: A modern, single-file format that embeds CRS information.
    -   **GeoJSON (`.geojson`)**: A lightweight, web-friendly format. Assumed to be in WGS84 (EPSG:4326) if no CRS is specified.

### 3.3 Coordinate Reference Systems (CRS)

The application supports a wide range of geographic and projected coordinate systems. It is **critical** to select the CRS that matches your input `X` and `Y` coordinates.

-   **Geographic**: WGS84 (EPSG:4326) for Latitude/Longitude data.
-   **Projected**: UTM (Northern & Southern hemispheres), Web Mercator, and regional systems like NAD83 and SIRGAS 2000.
-   **Automatic Suggestion**: The app analyzes your coordinate values and suggests a likely CRS to help you choose correctly.
-   **Validation**: After selection, the app validates if your coordinates fall within the typical range for that CRS.

## 4. Step-by-Step Workflow

Navigate through the steps using the sidebar menu.

### Step 1: Input Data

This page is for loading your dataset and defining its spatial context.

1.  **Upload Data File**: Click "Browse files" to upload your CSV or Excel file.
2.  **CSV Options (if applicable)**: If you upload a CSV, select the correct `Separator` (e.g., comma, semicolon).
3.  **Select Coordinate System**: Choose the CRS of your data from the dropdown. If your system isn't listed, select "Custom" and enter the EPSG code. Use the CRS suggestions as a guide.
4.  **Review Data**: Once loaded, a preview of the first 10 rows and a statistical summary will appear. Verify that the data has loaded correctly and that `X` and `Y` columns are present.

### Step 2: Exploratory Data Analysis

Here, you select the variables for analysis and handle redundancy.

1.  **Variable Selection**: In the multiselect box, choose the soil properties you want to include in the SQI calculation.
2.  **Correlation Analysis**:
    -   A **Correlation Matrix** heatmap is displayed, showing the relationships between your selected variables.
    -   Use the **slider** to set a correlation threshold (default: 0.98).
    -   Click **"Remove Highly Correlated Variables"**. The app will identify pairs of variables with a correlation above this threshold and automatically remove the one that has a higher average correlation with all other variables. This prevents multicollinearity issues in the PCA.
3.  **Review Final Variables**: The list of variables remaining after this step will be used for the rest of the analysis.

### Step 3: PCA Analysis

This step uses PCA to determine the objective weight of each soil variable.

1.  **Configure PCA**:
    -   **Scale Variables**: Keep this checked (default) to standardize variables before PCA. This is standard practice and highly recommended.
    -   **Eigenvalue Threshold**: Set the cutoff for determining significant principal components (PCs). The default of `1.0` (Kaiser's rule) is standard. Only PCs with an eigenvalue above this threshold will be used to calculate weights.
2.  **Run PCA**: Click **"Run PCA Analysis"**.
3.  **Interpret Results**:
    -   **Scree Plot**: This primary chart shows the eigenvalue for each PC. The red dashed line is your threshold. Look for the "elbow" in the plot to visually confirm the number of significant components.
    -   **Summary & Weights**: This tab shows the final calculated weights for each variable in a bar chart and provides key statistics about the PCA model.
    -   **Biplot**: This advanced plot visualizes both the samples (blue dots) and the variable loadings (red arrows) on two selected PCs. Longer arrows indicate a stronger influence of that variable on the components.
    -   **Loadings Heatmap**: This shows the contribution (loading) of each variable to each PC. Red indicates a positive correlation, and blue indicates a negative one.

### Step 4: SQI Calculation

Here, you score the indicators and calculate the final SQI.

1.  **Normalization Settings**: For each variable, you must define how it contributes to soil quality.
    -   `more_is_better`: Higher values get a higher score (e.g., Organic Matter).
    -   `minus_is_better`: Lower values get a higher score (e.g., Bulk Density).
    -   `range_optimo`: Values within a specified optimal range get a perfect score (1), and values outside get a zero (e.g., pH). You must provide the min and max for the optimal range.
    *The app provides sensible defaults for common indicators.*
2.  **Calculate SQI**: Click the **"Calculate SQI"** button. The app will:
    -   Normalize each variable according to your settings.
    -   Multiply each normalized score by its PCA-derived weight.
    -   Sum the weighted scores for each sample to get the final SQI value (ranging from 0 to 1).
3.  **Review Results**: A statistical summary and a histogram showing the distribution of the calculated SQI values will be displayed.

### Step 5: Spatial Assessment

This optional but powerful step creates a continuous map of soil quality from your sample points.

1.  **Upload Study Area (Optional)**: Upload a polygon file (`.zip`, `.gpkg`, `.geojson`) to define your study boundary. If provided, the final interpolated map will be clipped to this specific area.

2.  **Interpolation Settings**: This section controls how the spatial surface is generated.

    -   **Method**: Choose the algorithm used to estimate SQI values between your sample points.
        -   `Inverse Distance Weighting (IDW)`: This is a straightforward and fast deterministic method. It assumes that points closer to each other are more alike than those farther apart. The influence of a sample point on a grid cell is inversely proportional to its distance.
            -   **Power Parameter**: This setting (default: 2.0) controls the significance of surrounding points. A higher power value gives more weight to the nearest points, resulting in a less smooth, more detailed surface that closely honors the sample data. A lower value results in a smoother, more averaged surface.
        -   `Ordinary Kriging (Simplified)`: This option provides a more advanced interpolation that can capture more complex spatial patterns.
            -   **Note on Implementation**: While labeled "Ordinary Kriging" for simplicity, the application uses the `scipy.griddata` function with the `'cubic'` method. This is a deterministic cubic interpolation, not a full geostatistical kriging implementation. It fits a smooth, continuous polynomial surface through the data points and is excellent for creating visually appealing maps, but it does not provide the statistical error modeling of true Ordinary Kriging.
        -   `True Ordinary Kriging`: This option implements genuine geostatistical Ordinary Kriging using the PyKrige library.
            -   **Features**:
                - Automatic variogram modeling with multiple model options (spherical, exponential, gaussian, linear, power)
                - Statistical error estimation (kriging variance) for uncertainty quantification
                - Advanced parameters for variogram calculation
                - Variance visualization and export capabilities
            -   **Benefits**: Provides both interpolated values and prediction uncertainty, making it ideal for scientific applications where understanding confidence levels is important
            -   **Use Cases**: Recommended for research, environmental assessments, and applications requiring uncertainty quantification

    -   **Grid Cell Size**: This defines the spatial resolution of the output map in meters. A smaller value (e.g., 10m) will create a higher-resolution, more detailed map but will require significantly more computation time and memory. A larger value (e.g., 100m) will process much faster but will produce a coarser, more generalized map.

3.  **Run Interpolation**: After configuring your settings, click this button. The application will perform the following actions:
    -   It creates a regular grid of points (a "fishnet") covering the extent of your data or the uploaded study area.
    -   It uses the selected interpolation method (IDW or Cubic) to estimate an SQI value for every cell in the grid.
    -   If a study area was provided, it clips the interpolated grid, setting all values outside the polygon to null.
    -   The resulting grid is then used to generate the heatmap and contour overlays in the final step.

### Step 6: Results

This final page presents all results for visualization and download.

1.  **Spatial Distribution Map**: An interactive map displays:
    -   **Sample Points**: Your original data points, color-coded by their SQI value. Click on a point to see its details.
    -   **Study Area**: The boundary polygon you uploaded (if any).
    -   **Surface Overlays**: You can toggle two types of surface visualizations:
        -   **Interpolated Heatmap**: A continuous color gradient representing the interpolated SQI surface.
        -   **Contour Lines**: A representation of the surface using colored points that follow lines of equal SQI value.
    -   **Opacity Control**: Use the slider to adjust the transparency of the surface overlays.
2.  **Download Results**:
    -   **Download SQI Data (CSV)**: Downloads a CSV file containing your original data plus the calculated SQI for each sample point.
    -   **Export Interpolated Raster (GeoTIFF)**: This generates and downloads a georeferenced TIFF file of the interpolated SQI map. This file can be directly used in GIS software like QGIS or ArcGIS for further analysis.
3.  **SQI Classification**: A table summarizes the number and percentage of samples falling into different quality classes (e.g., Low, Medium, High).
4.  **Final Summary**: A text block provides a concise summary of the entire analysis.

## 5. Key Functions Explained

-   `load_polygon_file()`: Robustly handles the uploading and reading of different geospatial file formats (`.zip`, `.gpkg`, `.geojson`), including CRS detection.
-   `suggest_crs_from_coordinates()`: Analyzes the range and magnitude of X/Y coordinates to suggest the most likely CRS, reducing user error.
-   `validate_coordinates()`: Checks if the input coordinates are within a valid range for the selected CRS.
-   `norm_indicator()`: The core normalization engine. It takes a soil variable and a scoring method (`more_is_better`, `minus_is_better`, `range_optimo`) and returns a normalized score between 0 and 1.
-   `create_pca_biplot()` / `create_loadings_heatmap()`: Generate advanced, interactive Plotly visualizations for interpreting PCA results.
-   `add_heatmap_overlay_to_map()` / `add_contour_overlay_to_map()`: These functions take the gridded interpolation data and add it as a visual layer to the Folium map, including on-the-fly coordinate transformation to WGS84 for web display.

## 6. Core Libraries Used

-   **`streamlit`**: The core framework for building the interactive web application.
-   **`pandas`**, **`numpy`**: For data manipulation and numerical operations.
-   **`geopandas`**, **`shapely`**: For handling geospatial vector data (polygons, points) and performing coordinate transformations.
-   **`scikit-learn`**: Used for Principal Component Analysis (PCA) and data scaling.
-   **`scipy`**: Provides the interpolation algorithms (IDW, Kriging).
-   **`plotly`**: For creating interactive charts and plots (Scree plot, heatmaps, histograms).
-   **`folium`**, **`streamlit-folium`**: For creating and displaying the interactive results map.
-   **`rasterio`**: For creating and exporting the final georeferenced raster (GeoTIFF) file.

## 7. Troubleshooting

-   **Incorrect Map Display**: If your points appear in the wrong location (e.g., in the ocean), the most likely cause is an **incorrect CRS selection** in Step 1. Double-check that the selected CRS matches your source data's coordinate system.
-   **Error Loading Polygon**: If a shapefile fails to load, ensure the `.zip` file contains the essential components: `.shp`, `.shx`, and `.dbf`. A `.prj` file is also highly recommended for correct CRS detection.
-   **Interpolation Error**: Errors during interpolation can occur if there are too few data points or if the points are clustered in one area. Ensure you have a reasonable number of samples with good spatial distribution.
-   **No Significant PCs Found**: If the PCA analysis reports no significant components, it means no underlying factor had an eigenvalue greater than your threshold. You may need to lower the threshold or reconsider the variables included in the analysis.