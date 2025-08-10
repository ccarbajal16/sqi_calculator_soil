import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Point
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

_page_icon_image = "🌱"  # Reverted to simple emoji icon

# Configure page
st.set_page_config(
    page_title="Soil Quality Index Calculator",
    page_icon=_page_icon_image,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for green theme
st.markdown("""
<style>
    /* Main theme colors */
    .stApp {
        background-color: #f8fffe;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #e8f5e8;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background-color: #2d5a2d;
        border-color: #2d5a2d;
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1e3f1e;
        border-color: #1e3f1e;
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background-color: #f0f8f0;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2d5a2d;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #2d5a2d;
    }
    
    /* Plotly chart background */
    .js-plotly-plot {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'selected_vars' not in st.session_state:
    st.session_state.selected_vars = None
if 'final_vars' not in st.session_state:
    st.session_state.final_vars = None
if 'pca_result' not in st.session_state:
    st.session_state.pca_result = None
if 'pca_weights' not in st.session_state:
    st.session_state.pca_weights = None
if 'normalized_data' not in st.session_state:
    st.session_state.normalized_data = None
if 'sqi_values' not in st.session_state:
    st.session_state.sqi_values = None
if 'spatial_data' not in st.session_state:
    st.session_state.spatial_data = None
if 'interpolated_result' not in st.session_state:
    st.session_state.interpolated_result = None
if 'study_area_polygon' not in st.session_state:
    st.session_state.study_area_polygon = None

# Normalization function (equivalent to R's norm_indicator)
def load_polygon_file(uploaded_file):
    """
    Load polygon from various geospatial file formats
    Returns: (GeoDataFrame, success_message, error_message)
    """
    import json  # Import at function level to avoid scope issues

    try:
        file_name = uploaded_file.name.lower()

        if file_name.endswith('.zip'):
            # Handle zipped shapefile
            import zipfile
            import tempfile
            import os

            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract zip file
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Find the .shp file
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if not shp_files:
                    return None, None, "No .shp file found in the uploaded zip"

                # Check for required files
                base_name = shp_files[0][:-4]  # Remove .shp extension
                required_extensions = ['.shp', '.shx', '.dbf']
                missing_files = []

                for ext in required_extensions:
                    if not os.path.exists(os.path.join(temp_dir, base_name + ext)):
                        missing_files.append(ext)

                if missing_files:
                    return None, None, f"Missing required shapefile components: {', '.join(missing_files)}"

                # Load shapefile
                shp_path = os.path.join(temp_dir, shp_files[0])
                gdf = gpd.read_file(shp_path)

                return gdf, f"Shapefile loaded successfully ({len(gdf)} polygon(s))", None

        elif file_name.endswith('.gpkg'):
            # Handle GeoPackage
            import tempfile
            import os

            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Load GeoPackage
                gdf = gpd.read_file(tmp_path)

                # Check if it contains polygons
                geom_types = gdf.geometry.geom_type.unique()
                if not any(geom_type in ['Polygon', 'MultiPolygon'] for geom_type in geom_types):
                    return None, None, "GeoPackage does not contain polygon geometries"

                # Filter to only polygon geometries if mixed
                if len(geom_types) > 1:
                    gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

                return gdf, f"GeoPackage loaded successfully ({len(gdf)} polygon(s))", None

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        elif file_name.endswith('.geojson'):
            # Handle GeoJSON
            import io

            # Read and validate GeoJSON
            geojson_content = uploaded_file.read().decode('utf-8')

            # Validate JSON format
            try:
                json.loads(geojson_content)
            except json.JSONDecodeError as e:
                return None, None, f"Invalid GeoJSON format: {str(e)}"

            # Create BytesIO object from content for geopandas
            gdf = gpd.read_file(io.StringIO(geojson_content))

            # Check if it contains polygons
            geom_types = gdf.geometry.geom_type.unique()
            if not any(geom_type in ['Polygon', 'MultiPolygon'] for geom_type in geom_types):
                return None, None, "GeoJSON does not contain polygon geometries"

            # Filter to only polygon geometries if mixed
            if len(geom_types) > 1:
                gdf = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])]

            # GeoJSON typically uses WGS84, but check if CRS is specified
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')  # Assume WGS84 for GeoJSON

            return gdf, f"GeoJSON loaded successfully ({len(gdf)} polygon(s))", None

        else:
            return None, None, f"Unsupported file format: {file_name}"

    except json.JSONDecodeError as e:
        return None, None, f"Invalid GeoJSON format: {str(e)}"
    except Exception as e:
        return None, None, f"Error loading polygon file: {str(e)}"

def add_heatmap_overlay_to_map(folium_map, interpolated_result, current_crs, opacity=0.6):
    """
    Add interpolated surface as heatmap overlay to Folium map using grid points
    """
    try:
        from folium.plugins import HeatMap

        # Get interpolated data
        grid_x = interpolated_result['grid_x']
        grid_y = interpolated_result['grid_y']
        interpolated = interpolated_result['interpolated']

        # Create list of points for heatmap
        heat_data = []

        # Sample grid points for heatmap (reduce density for performance)
        step = max(1, min(grid_x.shape) // 50)  # Sample every nth point

        for i in range(0, grid_x.shape[0], step):
            for j in range(0, grid_x.shape[1], step):
                if not np.isnan(interpolated[i, j]) and interpolated[i, j] > 0:
                    x_coord = grid_x[i, j]
                    y_coord = grid_y[i, j]
                    sqi_value = interpolated[i, j]

                    # Transform coordinates to WGS84 if needed
                    if current_crs != 'EPSG:4326':
                        import geopandas as gpd
                        from shapely.geometry import Point

                        point_gdf = gpd.GeoDataFrame(
                            geometry=[Point(x_coord, y_coord)],
                            crs=current_crs
                        )
                        point_wgs84 = point_gdf.to_crs('EPSG:4326')

                        lat = point_wgs84.geometry.y.iloc[0]
                        lon = point_wgs84.geometry.x.iloc[0]
                    else:
                        lat = y_coord
                        lon = x_coord

                    # Add point with weight based on SQI value
                    heat_data.append([lat, lon, sqi_value])

        if len(heat_data) > 0:
            # Add heatmap to map
            HeatMap(
                heat_data,
                min_opacity=0.2,
                max_zoom=18,
                radius=15,
                blur=10,
                gradient={
                    0.0: 'red',
                    0.3: 'orange',
                    0.5: 'yellow',
                    0.7: 'lightgreen',
                    1.0: 'green'
                }
            ).add_to(folium_map)

            return True, f"Heatmap overlay added with {len(heat_data)} points"
        else:
            return False, "No valid interpolated points found for heatmap"

    except Exception as e:
        return False, f"Error adding heatmap overlay: {str(e)}"

def add_contour_overlay_to_map(folium_map, interpolated_result, current_crs):
    """
    Add simple contour representation using circles at grid points
    """
    try:
        # Get interpolated data
        grid_x = interpolated_result['grid_x']
        grid_y = interpolated_result['grid_y']
        interpolated = interpolated_result['interpolated']

        # Create contour levels
        valid_values = interpolated[~np.isnan(interpolated)]
        if len(valid_values) == 0:
            return False, "No valid interpolated values for contouring"

        min_val, max_val = valid_values.min(), valid_values.max()
        if max_val <= min_val:
            return False, "No variation in interpolated values for contouring"

        # Create 5 contour levels
        levels = np.linspace(min_val, max_val, 5)

        # Sample grid points for contour representation
        step = max(1, min(grid_x.shape) // 30)  # Sample every nth point
        contour_points = 0

        for i in range(0, grid_x.shape[0], step):
            for j in range(0, grid_x.shape[1], step):
                if not np.isnan(interpolated[i, j]):
                    x_coord = grid_x[i, j]
                    y_coord = grid_y[i, j]
                    sqi_value = interpolated[i, j]

                    # Find which contour level this point belongs to
                    level_idx = np.digitize(sqi_value, levels) - 1
                    level_idx = max(0, min(level_idx, len(levels) - 1))

                    # Transform coordinates to WGS84 if needed
                    if current_crs != 'EPSG:4326':
                        import geopandas as gpd
                        from shapely.geometry import Point

                        point_gdf = gpd.GeoDataFrame(
                            geometry=[Point(x_coord, y_coord)],
                            crs=current_crs
                        )
                        point_wgs84 = point_gdf.to_crs('EPSG:4326')

                        lat = point_wgs84.geometry.y.iloc[0]
                        lon = point_wgs84.geometry.x.iloc[0]
                    else:
                        lat = y_coord
                        lon = x_coord

                    # Color based on SQI level
                    if level_idx == 0:
                        color = 'red'
                        level_name = 'Very Low'
                    elif level_idx == 1:
                        color = 'orange'
                        level_name = 'Low'
                    elif level_idx == 2:
                        color = 'yellow'
                        level_name = 'Medium'
                    elif level_idx == 3:
                        color = 'lightgreen'
                        level_name = 'High'
                    else:
                        color = 'green'
                        level_name = 'Very High'

                    # Add small circle for contour point
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        popup=f"SQI: {sqi_value:.3f}<br>Level: {level_name}",
                        color=color,
                        weight=1,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(folium_map)

                    contour_points += 1

        return True, f"Contour representation added with {contour_points} points"

    except Exception as e:
        return False, f"Error adding contour overlay: {str(e)}"

def add_variance_overlay_to_map(folium_map, interpolated_result, current_crs, opacity=0.6):
    """
    Add kriging variance as heatmap overlay to Folium map using grid points
    """
    try:
        from folium.plugins import HeatMap

        # Get variance data
        grid_x = interpolated_result['grid_x']
        grid_y = interpolated_result['grid_y']
        variance_data = interpolated_result.get('variance')

        if variance_data is None:
            return False, "No variance data available"

        # Create list of points for variance heatmap
        variance_heat_data = []

        # Sample grid points for heatmap (reduce density for performance)
        step = max(1, min(grid_x.shape) // 50)  # Sample every nth point

        for i in range(0, grid_x.shape[0], step):
            for j in range(0, grid_x.shape[1], step):
                if not np.isnan(variance_data[i, j]) and variance_data[i, j] >= 0:
                    x_coord = grid_x[i, j]
                    y_coord = grid_y[i, j]
                    variance_value = variance_data[i, j]

                    # Transform coordinates to WGS84 if needed
                    if current_crs != 'EPSG:4326':
                        import geopandas as gpd
                        from shapely.geometry import Point

                        point_gdf = gpd.GeoDataFrame(
                            geometry=[Point(x_coord, y_coord)],
                            crs=current_crs
                        )
                        point_wgs84 = point_gdf.to_crs('EPSG:4326')

                        lat = point_wgs84.geometry.y.iloc[0]
                        lon = point_wgs84.geometry.x.iloc[0]
                    else:
                        lat = y_coord
                        lon = x_coord

                    # Add point with weight based on variance value
                    # Higher variance = higher uncertainty (more intense on heatmap)
                    variance_heat_data.append([lat, lon, variance_value])

        if len(variance_heat_data) > 0:
            # Create variance heatmap with different color scheme (blues for uncertainty)
            HeatMap(
                variance_heat_data,
                name='Kriging Variance (Uncertainty)',
                min_opacity=0.2,
                max_zoom=18,
                radius=15,
                blur=10,
                gradient={
                    0.0: 'lightblue',
                    0.3: 'blue',
                    0.6: 'darkblue',
                    0.8: 'navy',
                    1.0: 'purple'
                }
            ).add_to(folium_map)

            return True, f"Variance overlay added with {len(variance_heat_data)} points"
        else:
            return False, "No valid variance points found for heatmap"

    except Exception as e:
        return False, f"Error adding variance overlay: {str(e)}"

def suggest_crs_from_coordinates(df):
    """
    Suggest likely CRS based on coordinate values
    """
    x_vals = df['X'].values
    y_vals = df['Y'].values

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    suggestions = []

    # Check for geographic coordinates
    if (-180 <= x_min <= 180) and (-90 <= y_min <= 90) and (-180 <= x_max <= 180) and (-90 <= y_max <= 90):
        suggestions.append("EPSG:4326 (WGS84 Geographic)")

    # Check for UTM coordinates
    if (100000 <= x_min <= 900000) and (100000 <= x_max <= 900000):
        if y_min > 1000000:  # Likely Northern Hemisphere
            suggestions.append("UTM Northern Hemisphere (e.g., EPSG:32618 for Zone 18N)")
        elif y_max < 10000000:  # Likely Southern Hemisphere
            suggestions.append("UTM Southern Hemisphere (e.g., EPSG:32718 for Zone 18S)")

    # Check for Web Mercator
    if abs(x_min) > 1000000 and abs(x_max) > 1000000 and abs(y_min) > 1000000:
        suggestions.append("EPSG:3857 (Web Mercator)")

    return suggestions

def validate_coordinates(df, crs):
    """
    Validate if coordinates are reasonable for the given CRS
    """
    x_vals = df['X'].values
    y_vals = df['Y'].values

    if crs == "EPSG:4326":
        # Geographic coordinates: lon should be -180 to 180, lat should be -90 to 90
        if np.any((x_vals < -180) | (x_vals > 180)) or np.any((y_vals < -90) | (y_vals > 90)):
            return False, "Geographic coordinates out of valid range (lon: -180 to 180, lat: -90 to 90)"
        return True, "Geographic coordinates appear valid"

    elif crs == "EPSG:3857":  # Web Mercator
        # Web Mercator has specific bounds
        if np.any(np.abs(x_vals) > 20037508) or np.any(np.abs(y_vals) > 20048966):
            return False, "Web Mercator coordinates outside valid range"
        return True, "Web Mercator coordinates appear valid"

    elif "UTM" in crs or crs.startswith("EPSG:327") or crs.startswith("EPSG:326") or crs.startswith("EPSG:269") or crs.startswith("EPSG:319"):
        # UTM and similar projected coordinates
        if np.any(np.abs(x_vals) > 1e7) or np.any(np.abs(y_vals) > 1e8):
            return False, "Projected coordinates seem too large"

        # UTM X (Easting) should be roughly 160,000 to 840,000
        if np.any((x_vals < 100000) | (x_vals > 900000)):
            return False, "UTM Easting coordinates outside typical range (100,000 - 900,000)"

        # UTM Y (Northing) validation depends on hemisphere
        if "S" in crs or any(y < 1000000 for y in y_vals):  # Southern hemisphere or small northing values
            if np.any((y_vals < 1000000) | (y_vals > 10000000)):
                return False, "UTM Northing coordinates outside typical range for Southern Hemisphere"
        else:  # Northern hemisphere
            if np.any((y_vals < 1000000) | (y_vals > 9500000)):
                return False, "UTM Northing coordinates outside typical range for Northern Hemisphere"

        return True, "UTM coordinates appear valid"

    else:
        # Generic projected coordinate validation
        if np.any(np.abs(x_vals) > 1e8) or np.any(np.abs(y_vals) > 1e8):
            return False, "Projected coordinates seem too large"
        return True, "Projected coordinates appear reasonable"

def norm_indicator(x, tipo, min_optimo=None, max_optimo=None):
    """
    Normalize soil indicators based on different criteria
    """
    x = np.array(x)
    
    if tipo == "more_is_better":
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            return np.ones_like(x)
        normalized = (x - x_min) / (x_max - x_min)
        return normalized
    
    elif tipo == "minus_is_better":
        x_min, x_max = np.min(x), np.max(x)
        if x_max == x_min:
            return np.ones_like(x)
        normalized = (x_max - x) / (x_max - x_min)
        return normalized
    
    elif tipo == "range_optimo":
        if min_optimo is None or max_optimo is None:
            raise ValueError("min_optimo and max_optimo must be provided for range_optimo")
        normalized = np.where((x >= min_optimo) & (x <= max_optimo), 1, 0)
        return normalized
    
    else:
        raise ValueError("Type must be 'more_is_better', 'minus_is_better' or 'range_optimo'")

def create_pca_biplot(pca_result, variable_names, pc1=0, pc2=1, n_samples=100):
    """
    Create a PCA biplot showing both samples and variable loadings
    """
    # Get PCA components and transformed data
    components = pca_result['components']
    transformed_data = pca_result['transformed_data']
    explained_var = pca_result['explained_variance_ratio']
    
    # Sample data points for visualization (to avoid overcrowding)
    n_total = transformed_data.shape[0]
    if n_total > n_samples:
        sample_indices = np.random.choice(n_total, n_samples, replace=False)
        sample_data = transformed_data[sample_indices]
    else:
        sample_data = transformed_data
    
    # Create the biplot
    fig = go.Figure()
    
    # Add sample points
    fig.add_trace(go.Scatter(
        x=sample_data[:, pc1],
        y=sample_data[:, pc2],
        mode='markers',
        marker=dict(
            size=6,
            color='lightblue',
            opacity=0.6,
            line=dict(width=1, color='darkblue')
        ),
        name='Samples',
        hovertemplate='PC%d: %%{x:.2f}<br>PC%d: %%{y:.2f}<extra></extra>' % (pc1+1, pc2+1)
    ))
    
    # Add variable vectors (loadings)
    scale_factor = 3  # Scale factor for arrow visibility
    
    for i, var_name in enumerate(variable_names):
        # Get loadings for the selected PCs
        loading_x = components[pc1, i] * scale_factor
        loading_y = components[pc2, i] * scale_factor
        
        # Add arrow
        fig.add_trace(go.Scatter(
            x=[0, loading_x],
            y=[0, loading_y],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=[0, 8], color='red'),
            name=var_name,
            showlegend=False,
            hovertemplate=f'{var_name}<br>Loading PC{pc1+1}: {components[pc1, i]:.3f}<br>Loading PC{pc2+1}: {components[pc2, i]:.3f}<extra></extra>'
        ))
        
        # Add variable label
        fig.add_annotation(
            x=loading_x * 1.1,
            y=loading_y * 1.1,
            text=var_name,
            showarrow=False,
            font=dict(size=10, color='red'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='red',
            borderwidth=1
        )
    
    # Update layout
    fig.update_layout(
        title=f'PCA Biplot - PC{pc1+1} vs PC{pc2+1}',
        xaxis_title=f'PC{pc1+1} ({explained_var[pc1]:.1%} variance)',
        yaxis_title=f'PC{pc2+1} ({explained_var[pc2]:.1%} variance)',
        height=600,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1),
        yaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    )
    
    return fig

def create_loadings_heatmap(pca_result, variable_names, n_components=None):
    """
    Create a heatmap of PCA loadings
    """
    components = pca_result['components']
    explained_var = pca_result['explained_variance_ratio']
    
    if n_components is None:
        n_components = min(len(variable_names), 5)  # Show first 5 components
    
    # Create loadings matrix
    loadings_matrix = components[:n_components, :].T
    
    # Create component labels
    component_labels = [f'PC{i+1}\n({explained_var[i]:.1%})' for i in range(n_components)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=loadings_matrix,
        x=component_labels,
        y=variable_names,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Loading Value'),
        hoverongaps=False,
        hovertemplate='Variable: %{y}<br>Component: %{x}<br>Loading: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='PCA Loadings Heatmap',
        xaxis_title='Principal Components',
        yaxis_title='Variables',
        height=400,
        plot_bgcolor='white'
    )
    
    return fig

# Default ranges and normalization types
DEFAULT_RANGES = {
    'CEC': [12.0, 20.0], 
    'BD': [1.2, 1.4],
    'EC': [2.0, 15.0],
    'OM': [1.0, 3.0],
    'K': [300, 800],
    'Clay': [10, 25],
    'Silt': [30, 50],
    'pH': [6.5, 7.5],
    'P': [15, 45],
    'Sand': [40, 60]
}

DEFAULT_NORM_TYPES = {
    'CEC': "more_is_better", 
    'BD': "minus_is_better",
    'EC': "range_optimo",
    'OM': "more_is_better",
    'K': "more_is_better",
    'Clay': "range_optimo",
    'Silt': "range_optimo",
    'pH': "range_optimo",
    'P': "more_is_better",
    'Sand': "range_optimo"
}

# Sidebar navigation
st.sidebar.title("🌱 SQI Calculator")
page = st.sidebar.selectbox(
    "Navigation",
    ["Guide", "1. Input Data", "2. Exploratory Data", "3. PCA Analysis", 
     "4. SQI Calculation", "5. Spatial Assessment", "6. Results"]
)

# GUIDE PAGE
if page == "Guide":
    st.title("🌱 Soil Quality Index Calculator")
    st.markdown("---")
    
    st.markdown("""
    ## Welcome to the Soil Quality Index Calculator
    
    This application implements a comprehensive workflow for calculating and mapping soil quality indices 
    using Principal Component Analysis (PCA) and geostatistical methods.
    
    ### Workflow Steps:
    
    1. **Input Data**: Upload your soil data (CSV/Excel) with X, Y coordinates and soil properties
    2. **Exploratory Analysis**: Examine correlations and remove highly correlated variables (>0.98)
    3. **PCA Analysis**: Perform Principal Component Analysis to determine variable weights
    4. **SQI Calculation**: Define normalization parameters and calculate the Soil Quality Index
    5. **Spatial Assessment**: Upload study area polygon (optional) and apply geostatistical interpolation (Kriging or IDW)
    6. **Results**: Visualize spatial distribution, download maps and export interpolated rasters
    
    ### Data Requirements:
    - **Mandatory columns**: X, Y (spatial coordinates)
    - **Soil properties**: pH, OM, Clay, Silt, Sand, CEC, BD, EC, P, K (recommended)
    - **Format**: CSV or Excel files
    - **Coordinate system**: UTM or Geographic coordinates
    - **Study area (optional)**: Multiple formats supported:
      - Zipped Shapefile (.zip containing .shp, .shx, .dbf, .prj)
      - GeoPackage (.gpkg) - Modern single-file format
      - GeoJSON (.geojson) - Lightweight web-friendly format
    
    ### Key Features:
    - Interactive correlation analysis
    - Automatic PCA weight calculation
    - Three normalization methods (more_is_better, minus_is_better, range_optimo)
    - Multi-format study area polygon upload (Shapefile, GeoPackage, GeoJSON)
    - Spatial interpolation with Kriging and IDW
    - Interactive mapping and classification
    - Export capabilities for results, maps, and interpolated rasters (GeoTIFF)
    """)
    
    st.info("👈 Use the sidebar to navigate through the workflow steps.")

# INPUT DATA PAGE
elif page == "1. Input Data":
    st.title("📁 Data Input")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Data File")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="File must contain X, Y coordinates and soil property columns"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    # CSV options
                    st.subheader("CSV Options")
                    sep = st.selectbox("Separator", [',', ';', '\t'], index=0)
                    header = st.checkbox("Header", value=True)
                    
                    df = pd.read_csv(uploaded_file, sep=sep, header=0 if header else None)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.raw_data = df
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Validate coordinates
                if not all(col in df.columns for col in ['X', 'Y']):
                    st.error("❌ Error: X and Y coordinate columns are required!")
                else:
                    st.success("✅ X and Y coordinates found")

                    # Suggest CRS based on coordinate values
                    crs_suggestions = suggest_crs_from_coordinates(df)
                    if crs_suggestions:
                        st.info("💡 **CRS Suggestions based on your coordinate values:**")
                        for suggestion in crs_suggestions:
                            st.write(f"• {suggestion}")

                    # Validate coordinate values if CRS is selected
                    if hasattr(st.session_state, 'crs') and st.session_state.crs:
                        is_valid, message = validate_coordinates(df, st.session_state.crs)
                        if is_valid:
                            st.success(f"✅ {message}")
                        else:
                            st.warning(f"⚠️ {message}")
                            st.info("💡 Please verify your coordinate system selection matches your data")
                    
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Coordinate System")
        crs_options = {
            "WGS84 Geographic (Lat/Lon)": "EPSG:4326",
            # UTM Southern Hemisphere
            "UTM Zone 10S": "EPSG:32710",
            "UTM Zone 11S": "EPSG:32711",
            "UTM Zone 12S": "EPSG:32712",
            "UTM Zone 13S": "EPSG:32713",
            "UTM Zone 14S": "EPSG:32714",
            "UTM Zone 15S": "EPSG:32715",
            "UTM Zone 16S": "EPSG:32716",
            "UTM Zone 17S": "EPSG:32717",
            "UTM Zone 18S": "EPSG:32718",
            "UTM Zone 19S": "EPSG:32719",
            "UTM Zone 20S": "EPSG:32720",
            "UTM Zone 21S": "EPSG:32721",
            "UTM Zone 22S": "EPSG:32722",
            "UTM Zone 23S": "EPSG:32723",
            # UTM Northern Hemisphere
            "UTM Zone 10N": "EPSG:32610",
            "UTM Zone 11N": "EPSG:32611",
            "UTM Zone 12N": "EPSG:32612",
            "UTM Zone 13N": "EPSG:32613",
            "UTM Zone 14N": "EPSG:32614",
            "UTM Zone 15N": "EPSG:32615",
            "UTM Zone 16N": "EPSG:32616",
            "UTM Zone 17N": "EPSG:32617",
            "UTM Zone 18N": "EPSG:32618",
            "UTM Zone 19N": "EPSG:32619",
            "UTM Zone 20N": "EPSG:32620",
            "UTM Zone 21N": "EPSG:32621",
            "UTM Zone 22N": "EPSG:32622",
            "UTM Zone 23N": "EPSG:32623",
            # Common regional systems
            "Web Mercator": "EPSG:3857",
            "NAD83 / UTM Zone 10N": "EPSG:26910",
            "NAD83 / UTM Zone 11N": "EPSG:26911",
            "NAD83 / UTM Zone 12N": "EPSG:26912",
            "SIRGAS 2000 / UTM Zone 18S": "EPSG:31978",
            "SIRGAS 2000 / UTM Zone 19S": "EPSG:31979",
            "SIRGAS 2000 / UTM Zone 20S": "EPSG:31980",
            "Custom": "custom"
        }

        crs_choice = st.selectbox("Select CRS", list(crs_options.keys()), index=2)  # Default to UTM 18S

        if crs_choice == "Custom":
            custom_crs = st.text_input("Enter EPSG code", value="EPSG:32718")
            st.session_state.crs = custom_crs
        else:
            st.session_state.crs = crs_options[crs_choice]

        # Display current CRS info
        if hasattr(st.session_state, 'crs') and st.session_state.crs:
            st.info(f"📍 Current CRS: {st.session_state.crs}")
            if st.session_state.crs != "EPSG:4326":
                st.info("ℹ️ Coordinates will be transformed to WGS84 for web mapping")
    
    # Data preview
    if st.session_state.raw_data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.raw_data.head(10))
        
        st.subheader("Data Summary")
        st.dataframe(st.session_state.raw_data.describe())

# EXPLORATORY DATA PAGE
elif page == "2. Exploratory Data":
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    if st.session_state.raw_data is None:
        st.warning("⚠️ Please upload data first in the Input Data section.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Variable Selection")
            
            # Get numeric variables excluding coordinates
            numeric_vars = st.session_state.raw_data.select_dtypes(include=[np.number]).columns.tolist()
            numeric_vars = [var for var in numeric_vars if var not in ['X', 'Y']]
            
            if len(numeric_vars) == 0:
                st.error("No numeric variables found (excluding X, Y coordinates)")
            else:
                selected_vars = st.multiselect(
                    "Select variables for analysis:",
                    numeric_vars,
                    default=numeric_vars
                )
                st.session_state.selected_vars = selected_vars
                
                st.subheader("Correlation Analysis")
                cor_threshold = st.slider(
                    "Correlation threshold for removal:",
                    min_value=0.5, max_value=1.0, value=0.98, step=0.01
                )
                
                if st.button("Remove Highly Correlated Variables"):
                    if len(selected_vars) > 1:
                        # Calculate correlation matrix
                        corr_data = st.session_state.raw_data[selected_vars].corr()
                        
                        # Find highly correlated pairs
                        high_corr_pairs = []
                        for i in range(len(corr_data.columns)):
                            for j in range(i+1, len(corr_data.columns)):
                                if abs(corr_data.iloc[i, j]) > cor_threshold:
                                    high_corr_pairs.append((corr_data.columns[i], corr_data.columns[j]))
                        
                        # Remove variables with highest mean correlation
                        vars_to_remove = set()
                        for var1, var2 in high_corr_pairs:
                            mean_corr1 = abs(corr_data[var1]).mean()
                            mean_corr2 = abs(corr_data[var2]).mean()
                            if mean_corr1 > mean_corr2:
                                vars_to_remove.add(var1)
                            else:
                                vars_to_remove.add(var2)
                        
                        final_vars = [var for var in selected_vars if var not in vars_to_remove]
                        st.session_state.final_vars = final_vars
                        
                        if vars_to_remove:
                            st.success(f"✅ Removed {len(vars_to_remove)} highly correlated variables")
                            st.info(f"Removed variables: {', '.join(vars_to_remove)}")
                        else:
                            st.info("No highly correlated variables found")
                    else:
                        st.session_state.final_vars = selected_vars
        
        with col2:
            st.subheader("Correlation Matrix")
            
            if st.session_state.selected_vars and len(st.session_state.selected_vars) > 1:
                corr_matrix = st.session_state.raw_data[st.session_state.selected_vars].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Variable Correlation Matrix"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        # Variable statistics
        if st.session_state.selected_vars:
            st.subheader("Variable Statistics")
            stats_df = st.session_state.raw_data[st.session_state.selected_vars].describe()
            st.dataframe(stats_df)

# PCA ANALYSIS PAGE
elif page == "3. PCA Analysis":
    st.title("📊 Principal Component Analysis")
    st.markdown("---")
    
    if st.session_state.final_vars is None:
        st.warning("⚠️ Please complete the exploratory analysis first.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("PCA Controls")
            
            scale_data = st.checkbox("Scale Variables", value=True)
            eigenvalue_threshold = st.number_input(
                "Eigenvalue Threshold:", 
                min_value=0.1, max_value=5.0, value=1.0, step=0.1
            )
            
            if st.button("Run PCA Analysis", type="primary"):
                # Prepare data
                pca_data = st.session_state.raw_data[st.session_state.final_vars].dropna()
                
                if len(pca_data) == 0:
                    st.error("No complete cases found in the data")
                else:
                    # Scale data if requested
                    if scale_data:
                        scaler = StandardScaler()
                        pca_data_scaled = scaler.fit_transform(pca_data)
                    else:
                        pca_data_scaled = pca_data.values
                    
                    # Perform PCA
                    pca = PCA()
                    pca_result = pca.fit_transform(pca_data_scaled)
                    
                    # Store results
                    st.session_state.pca_result = {
                        'pca': pca,
                        'transformed_data': pca_result,
                        'eigenvalues': pca.explained_variance_,
                        'explained_variance_ratio': pca.explained_variance_ratio_,
                        'components': pca.components_
                    }
                    
                    # Calculate weights based on significant PCs
                    significant_pcs = np.where(pca.explained_variance_ > eigenvalue_threshold)[0]
                    
                    if len(significant_pcs) == 0:
                        st.error("No significant principal components found")
                    else:
                        # Calculate weights as sum of absolute loadings for significant PCs
                        weights = np.zeros(len(st.session_state.final_vars))
                        for pc_idx in significant_pcs:
                            weights += np.abs(pca.components_[pc_idx]) * pca.explained_variance_ratio_[pc_idx]
                        
                        # Normalize weights
                        weights = weights / np.sum(weights)
                        
                        st.session_state.pca_weights = dict(zip(st.session_state.final_vars, weights))
                        
                        st.success(f"✅ PCA completed! {len(significant_pcs)} significant components found.")
        
        with col2:
            st.subheader("Scree Plot & Variance Analysis")
            st.markdown("*Eigenvalues and cumulative variance explained by each component*")
            
            if st.session_state.pca_result is not None:
                eigenvalues = st.session_state.pca_result['eigenvalues']
                explained_var = st.session_state.pca_result['explained_variance_ratio']
                
                # Create subplot with two y-axes
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Eigenvalues
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(eigenvalues) + 1)),
                    y=eigenvalues,
                    mode='lines+markers',
                    name='Eigenvalues',
                    line=dict(color='#2d5a2d', width=3),
                    marker=dict(size=8)
                ), secondary_y=False)
                
                # Cumulative explained variance
                cumulative_var = np.cumsum(explained_var)
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(eigenvalues) + 1)),
                    y=cumulative_var * 100,
                    mode='lines+markers',
                    name='Cumulative Variance %',
                    line=dict(color='#8fbc8f', width=2, dash='dash'),
                    marker=dict(size=6)
                ), secondary_y=True)
                
                # Threshold line
                fig.add_hline(y=eigenvalue_threshold, line_dash="dash", 
                             line_color="red", annotation_text="Threshold")
                
                # Update layout
                fig.update_xaxes(title_text="Principal Component")
                fig.update_yaxes(title_text="Eigenvalue", secondary_y=False)
                fig.update_yaxes(title_text="Cumulative Variance (%)", secondary_y=True)
                
                fig.update_layout(
                    title="Scree Plot with Cumulative Variance",
                    height=400,
                    plot_bgcolor='white',
                    legend=dict(x=0.7, y=0.95)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced PCA Visualizations
        if st.session_state.pca_result is not None:
            st.markdown("---")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["📊 Summary & Weights", "🎯 Biplot", "🔥 Loadings Heatmap"])
            
            with tab1:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Variable Weights")
                    if st.session_state.pca_weights:
                        weights_df = pd.DataFrame(
                            list(st.session_state.pca_weights.items()),
                            columns=['Variable', 'Weight']
                        ).sort_values('Weight', ascending=False)
                        
                        # Create a bar chart for weights
                        fig_weights = px.bar(
                            weights_df, 
                            x='Weight', 
                            y='Variable',
                            orientation='h',
                            title='PCA-derived Variable Weights',
                            color='Weight',
                            color_continuous_scale='Greens'
                        )
                        fig_weights.update_layout(height=400, plot_bgcolor='white')
                        st.plotly_chart(fig_weights, use_container_width=True)
                
                with col2:
                    st.subheader("Component Statistics")
                    
                    # Create summary statistics table
                    n_components = len(eigenvalues)
                    significant_pcs = np.where(eigenvalues > eigenvalue_threshold)[0]
                    
                    summary_data = {
                        'Metric': [
                            'Total Components',
                            'Significant Components',
                            'Total Variance Explained (%)',
                            'Variance by Significant PCs (%)',
                            'Kaiser-Meyer-Olkin (approx)'
                        ],
                        'Value': [
                            str(n_components),
                            str(len(significant_pcs)),
                            f"{np.sum(explained_var)*100:.1f}",
                            f"{np.sum(explained_var[significant_pcs])*100:.1f}" if len(significant_pcs) > 0 else "0.0",
                            f"{np.mean(explained_var[:3]):.3f}"  # Approximation
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, hide_index=True)
                    
                    # Component details
                    st.subheader("Component Details")
                    comp_details = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(min(5, len(eigenvalues)))],
                        'Eigenvalue': eigenvalues[:5],
                        'Variance %': explained_var[:5] * 100,
                        'Cumulative %': np.cumsum(explained_var[:5]) * 100
                    })
                    st.dataframe(comp_details, hide_index=True)
            
            with tab2:
                st.subheader("PCA Biplot")
                st.markdown("""The biplot shows both sample scores (blue dots) and variable loadings (red arrows). 
                Longer arrows indicate variables with stronger influence on the principal components.""")
                
                # PC selection for biplot
                col1, col2 = st.columns([1, 1])
                with col1:
                    pc1_select = st.selectbox("X-axis (PC)", 
                                            options=list(range(1, min(6, len(eigenvalues)+1))), 
                                            index=0, key="pc1_biplot")
                with col2:
                    pc2_select = st.selectbox("Y-axis (PC)", 
                                            options=list(range(1, min(6, len(eigenvalues)+1))), 
                                            index=1, key="pc2_biplot")
                
                # Create and display biplot
                biplot_fig = create_pca_biplot(
                    st.session_state.pca_result, 
                    st.session_state.final_vars, 
                    pc1=pc1_select-1, 
                    pc2=pc2_select-1
                )
                st.plotly_chart(biplot_fig, use_container_width=True)
            
            with tab3:
                st.subheader("PCA Loadings Heatmap")
                st.markdown("""The heatmap shows how each variable contributes to each principal component. 
                Red indicates positive loadings, blue indicates negative loadings.""")
                
                # Number of components to show
                n_comp_show = st.slider("Number of components to display", 
                                      min_value=2, max_value=min(8, len(eigenvalues)), 
                                      value=min(5, len(eigenvalues)))
                
                # Create and display loadings heatmap
                loadings_fig = create_loadings_heatmap(
                    st.session_state.pca_result, 
                    st.session_state.final_vars, 
                    n_components=n_comp_show
                )
                st.plotly_chart(loadings_fig, use_container_width=True)
        
        # PCA Interpretation Guidelines
        if st.session_state.pca_result is not None:
            st.markdown("---")
            st.subheader("PCA Interpretation Guidelines")
            st.markdown("""
            - **Eigenvalues > 1.0** are typically considered significant components
            - **Scree Plot**: Look for the 'elbow' to determine optimal number of components  
            - **Biplot**: Longer arrows indicate variables with stronger influence on PCs
            - **Loadings**: Red/positive values contribute positively, blue/negative values contribute negatively
            - **Aim for 70-80%** cumulative variance explained for good data representation
            """)

# SQI CALCULATION PAGE
elif page == "4. SQI Calculation":
    st.title("🧮 SQI Calculation")
    st.markdown("---")
    
    if st.session_state.pca_weights is None:
        st.warning("⚠️ Please complete the PCA analysis first.")
    else:
        st.subheader("Normalization Settings")
        st.markdown("Configure normalization for each variable:")
        
        # Create normalization controls for each variable
        norm_settings = {}
        
        cols = st.columns(2)
        for i, var_name in enumerate(st.session_state.final_vars):
            with cols[i % 2]:
                st.markdown(f"**{var_name}**")
                
                # Default normalization type
                default_type = DEFAULT_NORM_TYPES.get(var_name, "more_is_better")
                norm_type = st.selectbox(
                    f"Normalization type for {var_name}:",
                    ["more_is_better", "minus_is_better", "range_optimo"],
                    index=["more_is_better", "minus_is_better", "range_optimo"].index(default_type),
                    key=f"norm_type_{var_name}"
                )
                
                norm_settings[var_name] = {'type': norm_type}
                
                # Range inputs for range_optimo
                if norm_type == "range_optimo":
                    default_range = DEFAULT_RANGES.get(var_name, [0, 1])
                    min_range = st.number_input(
                        f"Min optimal for {var_name}:",
                        value=float(default_range[0]),
                        key=f"min_range_{var_name}"
                    )
                    max_range = st.number_input(
                        f"Max optimal for {var_name}:",
                        value=float(default_range[1]),
                        key=f"max_range_{var_name}"
                    )
                    norm_settings[var_name]['min_range'] = min_range
                    norm_settings[var_name]['max_range'] = max_range
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Calculate SQI", type="primary"):
                try:
                    # Prepare data
                    sqi_data = st.session_state.raw_data[st.session_state.final_vars].dropna()
                    
                    # Normalize each variable
                    normalized_data = pd.DataFrame(index=sqi_data.index)
                    
                    for var_name in st.session_state.final_vars:
                        settings = norm_settings[var_name]
                        
                        if settings['type'] == "range_optimo":
                            normalized_values = norm_indicator(
                                sqi_data[var_name],
                                tipo=settings['type'],
                                min_optimo=settings['min_range'],
                                max_optimo=settings['max_range']
                            )
                        else:
                            normalized_values = norm_indicator(
                                sqi_data[var_name],
                                tipo=settings['type']
                            )
                        
                        normalized_data[var_name] = normalized_values
                    
                    st.session_state.normalized_data = normalized_data
                    
                    # Calculate weighted SQI
                    weights = np.array([st.session_state.pca_weights[var] for var in st.session_state.final_vars])
                    sqi_values = np.sum(normalized_data.values * weights, axis=1)
                    
                    st.session_state.sqi_values = sqi_values
                    
                    st.success("✅ SQI calculation completed successfully!")
                    
                    # SQI Statistics
                    st.subheader("SQI Statistics")
                    st.write(f"Mean: {np.mean(sqi_values):.3f}")
                    st.write(f"Std Dev: {np.std(sqi_values):.3f}")
                    st.write(f"Range: {np.min(sqi_values):.3f} - {np.max(sqi_values):.3f}")
                    
                except Exception as e:
                    st.error(f"❌ Error calculating SQI: {str(e)}")
        
        with col2:
            if st.session_state.sqi_values is not None:
                st.subheader("SQI Distribution")
                
                fig = px.histogram(
                    x=st.session_state.sqi_values,
                    nbins=20,
                    title="SQI Distribution",
                    labels={'x': 'SQI Values', 'y': 'Frequency'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# SPATIAL ASSESSMENT PAGE
elif page == "5. Spatial Assessment":
    st.title("🗺️ Spatial Assessment")
    st.markdown("---")
    
    if st.session_state.sqi_values is None:
        st.warning("⚠️ Please complete the SQI calculation first.")
    else:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Study Area (Optional)")

            # Polygon upload section
            uploaded_polygon = st.file_uploader(
                "Upload Study Area Polygon",
                type=['zip', 'gpkg', 'geojson'],
                help="Upload a polygon file to define the study area boundary:\n• ZIP: Zipped shapefile (.shp, .shx, .dbf, .prj)\n• GPKG: GeoPackage file (.gpkg)\n• GeoJSON: GeoJSON file (.geojson)"
            )

            if uploaded_polygon is not None:
                # Load polygon using the enhanced function
                polygon_gdf, success_msg, error_msg = load_polygon_file(uploaded_polygon)

                if error_msg:
                    st.error(f"❌ {error_msg}")
                elif polygon_gdf is not None:
                    st.session_state.study_area_polygon = polygon_gdf
                    st.success(f"✅ {success_msg}")

                    # Show polygon info
                    file_type = uploaded_polygon.name.split('.')[-1].upper()
                    st.write(f"**File Type:** {file_type}")
                    st.write(f"**Original CRS:** {polygon_gdf.crs}")
                    st.write(f"**Geometry Types:** {', '.join(polygon_gdf.geometry.geom_type.unique())}")
                    st.write(f"**Bounds:** {polygon_gdf.total_bounds}")

                    # Show attribute information if available
                    if len(polygon_gdf.columns) > 1:  # More than just geometry
                        attr_cols = [col for col in polygon_gdf.columns if col != 'geometry']
                        st.write(f"**Attributes:** {', '.join(attr_cols)}")

                    # Check CRS compatibility
                    if hasattr(st.session_state, 'crs') and st.session_state.crs:
                        if polygon_gdf.crs is None:
                            st.warning("⚠️ Polygon has no CRS information. Will assume same as data CRS during mapping.")
                        elif str(polygon_gdf.crs) != str(st.session_state.crs):
                            st.info(f"ℹ️ Polygon CRS ({polygon_gdf.crs}) differs from data CRS ({st.session_state.crs}). Will be transformed for mapping.")
                        else:
                            st.success("✅ Polygon CRS matches data CRS")
                    else:
                        st.info("ℹ️ Please select your data CRS to ensure proper coordinate alignment")

                    # Show format-specific information
                    if file_type == 'GEOJSON':
                        st.info("ℹ️ GeoJSON files typically use WGS84 coordinates (EPSG:4326)")
                    elif file_type == 'GPKG':
                        st.info("ℹ️ GeoPackage files store CRS information internally")
                    elif file_type == 'ZIP':
                        st.info("ℹ️ Shapefile CRS information loaded from .prj file")

            st.subheader("Interpolation Settings")

            interp_method = st.selectbox(
                "Interpolation Method:",
                ["Inverse Distance Weighting (IDW)", "Ordinary Kriging (Simplified)", "True Ordinary Kriging"]
            )

            if interp_method == "Inverse Distance Weighting (IDW)":
                power = st.number_input("Power Parameter:", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
            elif interp_method == "True Ordinary Kriging":
                st.subheader("Kriging Parameters")

                # Variogram model selection
                variogram_model = st.selectbox(
                    "Variogram Model:",
                    ["spherical", "exponential", "gaussian", "linear", "power"],
                    index=0,
                    help="Spherical is most commonly used for soil properties"
                )

                # Option to include variance estimation
                include_variance = st.checkbox(
                    "Include Variance Estimation",
                    value=True,
                    help="Calculate kriging variance (uncertainty) for each prediction"
                )

                # Advanced options in expander
                with st.expander("Advanced Kriging Options"):
                    nlags = st.number_input(
                        "Number of Lags for Variogram:",
                        min_value=6, max_value=20, value=12,
                        help="Number of lag distances for variogram calculation"
                    )

                    enable_plotting = st.checkbox(
                        "Enable Variogram Plotting",
                        value=False,
                        help="Display variogram plot (may slow down processing)"
                    )

            grid_size = st.number_input("Grid Cell Size (m):", min_value=10, max_value=1000, value=100, step=10)
            
            if st.button("Run Interpolation", type="primary"):
                try:
                    # Prepare spatial data
                    complete_rows = st.session_state.raw_data[st.session_state.final_vars].dropna().index
                    spatial_df = pd.DataFrame({
                        'X': st.session_state.raw_data.loc[complete_rows, 'X'],
                        'Y': st.session_state.raw_data.loc[complete_rows, 'Y'],
                        'SQI': st.session_state.sqi_values
                    })
                    
                    # Create prediction grid
                    if st.session_state.study_area_polygon is not None:
                        # Use polygon bounds for grid extent
                        bounds = st.session_state.study_area_polygon.total_bounds
                        x_min, y_min, x_max, y_max = bounds
                    else:
                        # Use data extent
                        x_min, x_max = spatial_df['X'].min(), spatial_df['X'].max()
                        y_min, y_max = spatial_df['Y'].min(), spatial_df['Y'].max()

                    grid_x = np.arange(x_min, x_max, grid_size)
                    grid_y = np.arange(y_min, y_max, grid_size)
                    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
                    
                    # Interpolation
                    if interp_method == "Inverse Distance Weighting (IDW)":
                        # Simple IDW implementation
                        points = spatial_df[['X', 'Y']].values
                        values = spatial_df['SQI'].values

                        grid_points = np.column_stack([grid_X.ravel(), grid_Y.ravel()])

                        # Calculate distances
                        distances = np.sqrt(((grid_points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))

                        # Avoid division by zero
                        distances = np.where(distances == 0, 1e-10, distances)

                        # IDW weights
                        weights = 1 / (distances ** power)
                        weights = weights / weights.sum(axis=1, keepdims=True)

                        # Interpolated values
                        interpolated_values = (weights * values).sum(axis=1)
                        interpolated_grid = interpolated_values.reshape(grid_X.shape)

                        # Store results without variance
                        kriging_variance = None

                    elif interp_method == "Ordinary Kriging (Simplified)":
                        # Simplified kriging using griddata (cubic interpolation)
                        points = spatial_df[['X', 'Y']].values
                        values = spatial_df['SQI'].values
                        grid_points = np.column_stack([grid_X.ravel(), grid_Y.ravel()])

                        # Simple kriging using griddata
                        interpolated_values = griddata(
                            points, values, grid_points, method='cubic', fill_value=np.nan
                        )
                        interpolated_grid = interpolated_values.reshape(grid_X.shape)

                        # Store results without variance
                        kriging_variance = None

                    else:  # True Ordinary Kriging
                        points = spatial_df[['X', 'Y']].values
                        values = spatial_df['SQI'].values

                        try:
                            # Create Ordinary Kriging object
                            OK = OrdinaryKriging(
                                points[:, 0], points[:, 1], values,
                                variogram_model=variogram_model,
                                nlags=nlags,
                                enable_plotting=enable_plotting,
                                verbose=False
                            )

                            # Perform kriging
                            if include_variance:
                                interpolated_values, variance_values = OK.execute(
                                    'grid', grid_X[0, :], grid_Y[:, 0], backend='vectorized'
                                )
                                kriging_variance = variance_values
                            else:
                                interpolated_values, _ = OK.execute(
                                    'grid', grid_X[0, :], grid_Y[:, 0], backend='vectorized'
                                )
                                kriging_variance = None

                            interpolated_grid = interpolated_values

                            st.success(f"✅ True Ordinary Kriging completed using {variogram_model} variogram model!")

                            # Display variogram information
                            if hasattr(OK, 'variogram_model_parameters'):
                                st.info(f"📊 Variogram parameters: {OK.variogram_model_parameters}")

                        except Exception as e:
                            st.error(f"❌ Error in True Ordinary Kriging: {str(e)}")
                            st.info("💡 Falling back to simplified kriging method...")

                            # Fallback to simplified method
                            grid_points = np.column_stack([grid_X.ravel(), grid_Y.ravel()])
                            interpolated_values = griddata(
                                points, values, grid_points, method='cubic', fill_value=np.nan
                            )
                            interpolated_grid = interpolated_values.reshape(grid_X.shape)
                            kriging_variance = None

                    # Clip to study area polygon if provided
                    if st.session_state.study_area_polygon is not None:
                        from shapely.geometry import Point

                        # Create mask for points inside polygon
                        polygon_geom = st.session_state.study_area_polygon.geometry.iloc[0]  # Use first polygon

                        # Create boolean mask for grid points
                        mask = np.zeros_like(interpolated_grid, dtype=bool)

                        for i in range(grid_X.shape[0]):
                            for j in range(grid_X.shape[1]):
                                point = Point(grid_X[i, j], grid_Y[i, j])
                                if polygon_geom.contains(point):
                                    mask[i, j] = True

                        # Apply mask to interpolated grid
                        interpolated_grid_clipped = np.where(mask, interpolated_grid, np.nan)
                    else:
                        interpolated_grid_clipped = interpolated_grid

                    # Store results
                    st.session_state.interpolated_result = {
                        'grid_x': grid_X,
                        'grid_y': grid_Y,
                        'interpolated': interpolated_grid_clipped,
                        'interpolated_full': interpolated_grid,  # Keep unclipped version
                        'method': interp_method,
                        'spatial_df': spatial_df,
                        'grid_size': grid_size,
                        'clipped': st.session_state.study_area_polygon is not None,
                        'variance': kriging_variance,  # Store kriging variance if available
                        'has_variance': kriging_variance is not None
                    }
                    
                    st.success("✅ Spatial interpolation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error in interpolation: {str(e)}")
        
        with col2:
            if st.session_state.interpolated_result is not None:
                st.subheader("Interpolation Summary")
                result = st.session_state.interpolated_result
                
                st.write(f"**Method:** {result['method']}")
                st.write(f"**Grid size:** {grid_size} m")
                st.write(f"**Grid dimensions:** {result['grid_x'].shape}")
                st.write(f"**Sample points:** {len(result['spatial_df'])}")

                valid_values = result['interpolated'][~np.isnan(result['interpolated'])]
                if len(valid_values) > 0:
                    st.write(f"**Interpolated SQI range:** {np.min(valid_values):.3f} - {np.max(valid_values):.3f}")

                # Show variance information if available
                if result.get('has_variance', False) and result.get('variance') is not None:
                    variance_data = result['variance']
                    valid_variance = variance_data[~np.isnan(variance_data)]
                    if len(valid_variance) > 0:
                        st.write(f"**Kriging Variance range:** {np.min(valid_variance):.6f} - {np.max(valid_variance):.6f}")
                        st.write(f"**Mean Kriging Standard Error:** {np.sqrt(np.mean(valid_variance)):.3f}")
                        st.info("📊 Kriging variance represents prediction uncertainty - lower values indicate higher confidence")

# RESULTS PAGE
elif page == "6. Results":
    st.title("📈 Results")
    st.markdown("---")
    
    if st.session_state.interpolated_result is None:
        st.warning("⚠️ Please complete the spatial assessment first.")
    else:
        # Create interactive map
        st.subheader("Spatial Distribution Map")

        # Show coordinate system info
        if hasattr(st.session_state, 'crs') and st.session_state.crs:
            st.info(f"📍 Data CRS: {st.session_state.crs} | Map CRS: WGS84 (EPSG:4326)")

        result = st.session_state.interpolated_result
        spatial_df = result['spatial_df']

        # Transform coordinates to WGS84 for web mapping
        try:
            # Get the current CRS
            current_crs = getattr(st.session_state, 'crs', 'EPSG:32718')

            # Create GeoDataFrame with original CRS
            gdf_points = gpd.GeoDataFrame(
                spatial_df,
                geometry=gpd.points_from_xy(spatial_df['X'], spatial_df['Y']),
                crs=current_crs
            )

            # Check if transformation is needed
            if current_crs == 'EPSG:4326':
                # Data is already in WGS84
                spatial_df_wgs84 = spatial_df.copy()
                spatial_df_wgs84['lon'] = spatial_df['X']
                spatial_df_wgs84['lat'] = spatial_df['Y']
                st.info("📍 Data is already in WGS84 - no transformation needed")
            else:
                # Transform to WGS84 for web mapping
                gdf_points_wgs84 = gdf_points.to_crs('EPSG:4326')

                # Extract transformed coordinates
                spatial_df_wgs84 = spatial_df.copy()
                spatial_df_wgs84['lon'] = gdf_points_wgs84.geometry.x
                spatial_df_wgs84['lat'] = gdf_points_wgs84.geometry.y
                st.success(f"✅ Coordinates transformed from {current_crs} to WGS84")

            # Validate transformed coordinates
            if np.any((spatial_df_wgs84['lon'] < -180) | (spatial_df_wgs84['lon'] > 180)) or \
               np.any((spatial_df_wgs84['lat'] < -90) | (spatial_df_wgs84['lat'] > 90)):
                st.error("❌ Transformed coordinates are outside valid geographic range. Please check your CRS selection.")
                st.stop()

            # Create folium map with proper center
            center_lat = spatial_df_wgs84['lat'].mean()
            center_lon = spatial_df_wgs84['lon'].mean()

            # Determine appropriate zoom level based on data extent
            lat_range = spatial_df_wgs84['lat'].max() - spatial_df_wgs84['lat'].min()
            lon_range = spatial_df_wgs84['lon'].max() - spatial_df_wgs84['lon'].min()
            max_range = max(lat_range, lon_range)

            if max_range > 10:
                zoom_start = 6
            elif max_range > 1:
                zoom_start = 10
            elif max_range > 0.1:
                zoom_start = 12
            else:
                zoom_start = 14

            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

            # Add interpolated surface overlay
            st.subheader("Surface Visualization Options")

            # Check if variance is available
            has_variance = (st.session_state.interpolated_result is not None and
                          st.session_state.interpolated_result.get('has_variance', False))

            if has_variance:
                overlay_options = st.columns(4)
            else:
                overlay_options = st.columns(3)

            with overlay_options[0]:
                show_heatmap = st.checkbox("Show Interpolated Heatmap", value=True)
            with overlay_options[1]:
                show_contours = st.checkbox("Show Contour Lines", value=False)
            with overlay_options[2]:
                overlay_opacity = st.slider("Overlay Opacity", 0.1, 1.0, 0.6, 0.1)

            # Add variance option if available
            if has_variance:
                with overlay_options[3]:
                    show_variance = st.checkbox("Show Kriging Variance", value=False,
                                              help="Display prediction uncertainty from True Ordinary Kriging")
            else:
                show_variance = False

            # Add heatmap overlay if requested
            if show_heatmap and st.session_state.interpolated_result is not None:
                heatmap_success, heatmap_msg = add_heatmap_overlay_to_map(
                    m, st.session_state.interpolated_result, current_crs, overlay_opacity
                )
                if heatmap_success:
                    st.success(f"✅ {heatmap_msg}")
                else:
                    st.warning(f"⚠️ {heatmap_msg}")

            # Add contour overlay if requested
            if show_contours and st.session_state.interpolated_result is not None:
                contour_success, contour_msg = add_contour_overlay_to_map(
                    m, st.session_state.interpolated_result, current_crs
                )
                if contour_success:
                    st.success(f"✅ {contour_msg}")
                else:
                    st.warning(f"⚠️ {contour_msg}")

            # Add variance overlay if requested
            if show_variance and st.session_state.interpolated_result is not None:
                variance_success, variance_msg = add_variance_overlay_to_map(
                    m, st.session_state.interpolated_result, current_crs, overlay_opacity
                )
                if variance_success:
                    st.success(f"✅ {variance_msg}")
                else:
                    st.warning(f"⚠️ {variance_msg}")

            # Add sample points with transformed coordinates
            for idx, row in spatial_df_wgs84.iterrows():
                # Color based on SQI value
                if row['SQI'] < 0.1:
                    color = 'red'
                    sqi_class = 'Very Low'
                elif row['SQI'] < 0.3:
                    color = 'orange'
                    sqi_class = 'Low'
                elif row['SQI'] < 0.5:
                    color = 'yellow'
                    sqi_class = 'Medium'
                else:
                    color = 'green'
                    sqi_class = 'High'

                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=6,
                    popup=f"SQI: {row['SQI']:.3f}<br>Class: {sqi_class}<br>UTM X: {row['X']:.0f}<br>UTM Y: {row['Y']:.0f}",
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)

            # Add study area polygon if available
            if st.session_state.study_area_polygon is not None:
                try:
                    polygon_gdf = st.session_state.study_area_polygon.copy()

                    # Check if polygon needs transformation
                    if polygon_gdf.crs is None:
                        st.warning("⚠️ Study area polygon has no CRS information. Assuming same as data CRS.")
                        polygon_gdf = polygon_gdf.set_crs(current_crs)

                    # Transform polygon to WGS84 if needed
                    if polygon_gdf.crs != 'EPSG:4326':
                        polygon_wgs84 = polygon_gdf.to_crs('EPSG:4326')
                        st.info(f"📍 Study area transformed from {polygon_gdf.crs} to WGS84")
                    else:
                        polygon_wgs84 = polygon_gdf
                        st.info("📍 Study area is already in WGS84")

                    # Add polygon to map
                    for idx, row in polygon_wgs84.iterrows():
                        # Convert polygon to GeoJSON-like format for folium
                        geom = row.geometry
                        if geom.geom_type == 'Polygon':
                            # Extract coordinates and ensure proper lat/lon order for folium
                            coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                            folium.Polygon(
                                locations=coords,
                                color='blue',
                                weight=3,
                                fillColor='lightblue',
                                fillOpacity=0.3,
                                popup=f"Study Area Boundary<br>Original CRS: {polygon_gdf.crs}"
                            ).add_to(m)
                        elif geom.geom_type == 'MultiPolygon':
                            for poly_idx, poly in enumerate(geom.geoms):
                                coords = [[lat, lon] for lon, lat in poly.exterior.coords]
                                folium.Polygon(
                                    locations=coords,
                                    color='blue',
                                    weight=3,
                                    fillColor='lightblue',
                                    fillOpacity=0.3,
                                    popup=f"Study Area Boundary {poly_idx + 1}<br>Original CRS: {polygon_gdf.crs}"
                                ).add_to(m)

                    st.success("✅ Study area polygon added to map")

                except Exception as e:
                    st.error(f"❌ Could not display study area polygon on map: {str(e)}")
                    st.info("💡 Please check that the polygon CRS matches your data CRS")

        except Exception as e:
            st.error(f"❌ Error creating map: {str(e)}")
            st.info("Using original coordinates as fallback...")

            # Fallback: assume coordinates are already in WGS84
            center_lat = spatial_df['Y'].mean()
            center_lon = spatial_df['X'].mean()

            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

            # Add sample points
            for idx, row in spatial_df.iterrows():
                # Color based on SQI value
                if row['SQI'] < 0.1:
                    color = 'red'
                    sqi_class = 'Very Low'
                elif row['SQI'] < 0.3:
                    color = 'orange'
                    sqi_class = 'Low'
                elif row['SQI'] < 0.5:
                    color = 'yellow'
                    sqi_class = 'Medium'
                else:
                    color = 'green'
                    sqi_class = 'High'

                folium.CircleMarker(
                    location=[row['Y'], row['X']],
                    radius=6,
                    popup=f"SQI: {row['SQI']:.3f}<br>Class: {sqi_class}",
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.8
                ).add_to(m)
        
        # Add layer control if overlays were added
        if (show_heatmap or show_contours or show_variance) and st.session_state.interpolated_result is not None:
            folium.LayerControl().add_to(m)

        # Display map
        map_data = st_folium(m, width=700, height=500)

        # Add legend for surface visualization
        if (show_heatmap or show_contours or show_variance) and st.session_state.interpolated_result is not None:
            st.subheader("Surface Visualization Legend")

            # Adjust columns based on what's displayed
            num_legends = sum([show_heatmap, show_contours, show_variance])
            legend_cols = st.columns(max(2, num_legends))

            with legend_cols[0]:
                if show_heatmap:
                    st.markdown("""
                    **Heatmap Colors:**
                    - 🔴 Red: Very Low SQI (0.0 - 0.3)
                    - 🟠 Orange: Low SQI (0.3 - 0.5)
                    - 🟡 Yellow: Medium SQI (0.5 - 0.7)
                    - 🟢 Light Green: High SQI (0.7 - 0.9)
                    - 🟢 Green: Very High SQI (0.9 - 1.0)
                    """)

            legend_idx = 1
            if len(legend_cols) > legend_idx:
                with legend_cols[legend_idx]:
                    if show_contours:
                        st.markdown("""
                        **Contour Points:**
                        - Small circles represent interpolated values
                        - Colors indicate SQI quality levels
                        - Click on points for exact SQI values
                        """)
                        legend_idx += 1

            if show_variance and len(legend_cols) > legend_idx:
                with legend_cols[legend_idx]:
                    st.markdown("""
                    **Kriging Variance (Uncertainty):**
                    - 🔵 Light Blue: Low uncertainty (high confidence)
                    - 🔵 Blue: Medium uncertainty
                    - 🔵 Dark Blue: High uncertainty
                    - 🟣 Purple: Very high uncertainty (low confidence)
                    - Higher intensity = greater prediction uncertainty
                    """)

                # Show interpolation statistics
                if st.session_state.interpolated_result is not None:
                    interpolated = st.session_state.interpolated_result['interpolated']
                    valid_values = interpolated[~np.isnan(interpolated)]
                    if len(valid_values) > 0:
                        st.markdown(f"""
                        **Interpolated Surface Statistics:**
                        - Min SQI: {valid_values.min():.3f}
                        - Max SQI: {valid_values.max():.3f}
                        - Mean SQI: {valid_values.mean():.3f}
                        - Grid cells: {len(valid_values):,}
                        """)
        
        # Results summary and downloads
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Download Results")

            # Prepare download data
            complete_rows = st.session_state.raw_data[st.session_state.final_vars].dropna().index
            result_data = st.session_state.raw_data.loc[complete_rows].copy()
            result_data['SQI'] = st.session_state.sqi_values

            # Download CSV
            csv = result_data.to_csv(index=False)
            st.download_button(
                label="📥 Download SQI Data (CSV)",
                data=csv,
                file_name=f"sqi_results_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

            # Download interpolated raster
            if st.button("📥 Export Interpolated Raster (GeoTIFF)", type="secondary"):
                try:
                    result = st.session_state.interpolated_result

                    # Create transform for the raster
                    grid_x = result['grid_x']
                    grid_y = result['grid_y']
                    interpolated = result['interpolated']

                    # Calculate transform - fix Y-axis orientation
                    x_min, x_max = grid_x.min(), grid_x.max()
                    y_min, y_max = grid_y.min(), grid_y.max()
                    
                    # Flip the interpolated array vertically to match GeoTIFF convention
                    # GeoTIFF Y-axis goes from top to bottom, but our grid goes from bottom to top
                    interpolated_flipped = np.flipud(interpolated)

                    transform = from_bounds(
                        x_min, y_min, x_max, y_max,
                        interpolated_flipped.shape[1], interpolated_flipped.shape[0]
                    )

                    # Create in-memory raster
                    with io.BytesIO() as buffer:
                        with rasterio.open(
                            buffer,
                            'w',
                            driver='GTiff',
                            height=interpolated_flipped.shape[0],
                            width=interpolated_flipped.shape[1],
                            count=1,
                            dtype=interpolated_flipped.dtype,
                            crs=st.session_state.crs if hasattr(st.session_state, 'crs') else 'EPSG:4326',
                            transform=transform,
                            compress='lzw'
                        ) as dst:
                            dst.write(interpolated_flipped, 1)

                        # Get the buffer content
                        buffer.seek(0)
                        raster_data = buffer.read()

                    # Download button for raster
                    st.download_button(
                        label="💾 Download GeoTIFF",
                        data=raster_data,
                        file_name=f"sqi_interpolated_{datetime.now().strftime('%Y%m%d')}.tif",
                        mime="application/octet-stream"
                    )

                    st.success("✅ Raster export ready for download!")

                except Exception as e:
                    st.error(f"❌ Error exporting raster: {str(e)}")

            # Variance export for True Ordinary Kriging
            if (st.session_state.interpolated_result is not None and
                st.session_state.interpolated_result.get('has_variance', False)):

                st.subheader("Kriging Variance Export")

                if st.button("📥 Export Variance Raster (GeoTIFF)", type="secondary"):
                    try:
                        result = st.session_state.interpolated_result
                        variance_data = result['variance']

                        if variance_data is not None:
                            # Create transform for the raster
                            grid_x = result['grid_x']
                            grid_y = result['grid_y']

                            # Calculate transform
                            x_min, x_max = grid_x.min(), grid_x.max()
                            y_min, y_max = grid_y.min(), grid_y.max()

                            # Flip the variance array vertically to match GeoTIFF convention
                            variance_flipped = np.flipud(variance_data)

                            transform = from_bounds(
                                x_min, y_min, x_max, y_max,
                                variance_flipped.shape[1], variance_flipped.shape[0]
                            )

                            # Create in-memory raster
                            with io.BytesIO() as buffer:
                                with rasterio.open(
                                    buffer,
                                    'w',
                                    driver='GTiff',
                                    height=variance_flipped.shape[0],
                                    width=variance_flipped.shape[1],
                                    count=1,
                                    dtype=variance_flipped.dtype,
                                    crs=st.session_state.crs if hasattr(st.session_state, 'crs') else 'EPSG:4326',
                                    transform=transform,
                                    compress='lzw'
                                ) as dst:
                                    dst.write(variance_flipped, 1)

                                # Get the buffer content
                                buffer.seek(0)
                                variance_raster_data = buffer.read()

                            # Download button for variance raster
                            st.download_button(
                                label="💾 Download Variance GeoTIFF",
                                data=variance_raster_data,
                                file_name=f"sqi_variance_{datetime.now().strftime('%Y%m%d')}.tif",
                                mime="application/octet-stream"
                            )

                            st.success("✅ Variance raster export ready for download!")

                    except Exception as e:
                        st.error(f"❌ Error creating variance raster: {str(e)}")

        with col2:
            st.subheader("SQI Classification")
            
            # Create classification table
            sqi_values = st.session_state.sqi_values
            
            classifications = []
            for sqi in sqi_values:
                if sqi < 0.1:
                    classifications.append('Very Low')
                elif sqi < 0.3:
                    classifications.append('Low')
                elif sqi < 0.5:
                    classifications.append('Medium')
                else:
                    classifications.append('High')
            
            class_counts = pd.Series(classifications).value_counts()
            class_df = pd.DataFrame({
                'Class': class_counts.index,
                'Count': class_counts.values,
                'Percentage': (class_counts.values / len(classifications) * 100).round(1)
            })
            
            st.dataframe(class_df)
        
        # Final summary
        st.subheader("Final Results Summary")
        
        summary_text = f"""
        **SOIL QUALITY INDEX ANALYSIS SUMMARY**
        
        **Dataset Information:**
        - Total samples: {len(st.session_state.sqi_values)}
        - Variables analyzed: {len(st.session_state.final_vars)}
        - Interpolation method: {result['method']}
        
        **SQI Statistics:**
        - Mean SQI: {np.mean(st.session_state.sqi_values):.3f}
        - Standard deviation: {np.std(st.session_state.sqi_values):.3f}
        - Range: {np.min(st.session_state.sqi_values):.3f} - {np.max(st.session_state.sqi_values):.3f}
        
        **Variables and Weights:**
        """
        
        for var_name in st.session_state.final_vars:
            weight = st.session_state.pca_weights[var_name]
            summary_text += f"\n- {var_name}: {weight:.4f}"
        
        st.text(summary_text)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🌱 **SQI Calculator v1.0**")
st.sidebar.markdown("Built with Streamlit")
