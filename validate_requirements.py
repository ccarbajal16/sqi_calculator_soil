#!/usr/bin/env python3
"""
Validation script to check if all required libraries are properly installed
and can be imported successfully.
"""

import sys
import importlib
from packaging import version

def check_import(module_name, min_version=None, package_name=None):
    """Check if a module can be imported and optionally check version"""
    try:
        module = importlib.import_module(module_name)
        
        if min_version and hasattr(module, '__version__'):
            current_version = module.__version__
            if version.parse(current_version) < version.parse(min_version):
                print(f"⚠️  {module_name}: {current_version} (minimum {min_version} required)")
                return False
            else:
                print(f"✅ {module_name}: {current_version}")
        else:
            print(f"✅ {module_name}: imported successfully")
        
        return True
        
    except ImportError as e:
        package = package_name or module_name
        print(f"❌ {module_name}: NOT FOUND - install with 'pip install {package}'")
        return False
    except Exception as e:
        print(f"❌ {module_name}: ERROR - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("SQI CALCULATOR - REQUIREMENTS VALIDATION")
    print("=" * 60)
    
    # Define required modules with minimum versions and package names
    requirements = [
        # Core web framework
        ("streamlit", "1.28.0"),
        
        # Data manipulation
        ("pandas", "1.5.0"),
        ("numpy", "1.20.0"),
        
        # Scientific computing
        ("scipy", "1.9.0"),
        ("sklearn", "1.1.0", "scikit-learn"),
        
        # NEW: Geostatistical interpolation
        ("pykrige", "1.7.0"),
        
        # Geospatial
        ("geopandas", "0.12.0"),
        ("shapely", "1.8.0"),
        ("rasterio", "1.3.0"),
        ("pyproj", "3.4.0"),
        ("fiona", "1.8.0"),
        
        # Visualization
        ("plotly", "5.10.0"),
        ("matplotlib", "3.5.0"),
        
        # Mapping
        ("folium", "0.14.0"),
        ("streamlit_folium", "0.11.0", "streamlit-folium"),
        
        # File support
        ("openpyxl", "3.0.0"),
        ("xlrd", "2.0.0"),
        ("PIL", "9.0.0", "pillow"),
    ]
    
    print(f"Python version: {sys.version}")
    print("-" * 60)
    
    success_count = 0
    total_count = len(requirements)
    
    for req in requirements:
        if len(req) == 2:
            module_name, min_version = req
            package_name = None
        else:
            module_name, min_version, package_name = req
            
        if check_import(module_name, min_version, package_name):
            success_count += 1
    
    print("-" * 60)
    print(f"Results: {success_count}/{total_count} modules successfully validated")
    
    if success_count == total_count:
        print("🎉 ALL REQUIREMENTS SATISFIED!")
        print("\nYou can now run the application with:")
        print("streamlit run sqi_app.py")
        
        # Test specific kriging functionality
        print("\n" + "=" * 60)
        print("TESTING TRUE ORDINARY KRIGING")
        print("=" * 60)
        
        try:
            from pykrige.ok import OrdinaryKriging
            import numpy as np
            
            # Quick test
            x = np.array([0, 1, 2, 3, 4])
            y = np.array([0, 1, 2, 3, 4])
            z = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            
            OK = OrdinaryKriging(x, y, z, variogram_model='spherical', verbose=False)
            z_pred, variance = OK.execute('grid', [0.5, 1.5], [0.5, 1.5], backend='vectorized')
            
            print("✅ True Ordinary Kriging test successful!")
            print(f"   Predicted values: {z_pred.flatten()}")
            print(f"   Variance values: {variance.flatten()}")
            
        except Exception as e:
            print(f"❌ True Ordinary Kriging test failed: {str(e)}")
            
    else:
        print("❌ SOME REQUIREMENTS MISSING!")
        print("\nTo install missing packages, run:")
        print("pip install -r requirements.txt")
        
        if success_count < total_count * 0.5:
            print("\nFor easier installation, consider using conda:")
            print("conda install -c conda-forge streamlit pandas numpy scipy scikit-learn")
            print("conda install -c conda-forge geopandas plotly folium rasterio")
            print("pip install streamlit-folium pykrige")
    
    print("\n" + "=" * 60)
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
