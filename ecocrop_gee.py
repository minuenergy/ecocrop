"""
EcoCrop Crop Suitability Assessment using Google Earth Engine

This script uses GEE to fetch climate data (temperature, precipitation) and
evaluates crop suitability based on FAO EcoCrop database parameters.

Requirements:
    pip install earthengine-api pandas numpy matplotlib folium geemap

Usage:
    python ecocrop_gee.py

Author: Based on EcoCrop methodology
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import os


# =============================================================================
# GEE Initialization
# =============================================================================

def initialize_gee(project: Optional[str] = None):
    """
    Initialize Google Earth Engine.

    Args:
        project: GEE project ID (optional, required for newer authentication)
    """
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("✓ Google Earth Engine initialized successfully")
    except Exception as e:
        print("Authenticating with Google Earth Engine...")
        ee.Authenticate()
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        print("✓ Google Earth Engine initialized successfully")


# =============================================================================
# EcoCrop Database Functions
# =============================================================================

def load_ecocrop_database(csv_path: str = None) -> pd.DataFrame:
    """
    Load EcoCrop database from CSV file.

    Args:
        csv_path: Path to EcoCrop CSV file. If None, uses default path.

    Returns:
        DataFrame with crop parameters
    """
    if csv_path is None:
        # Try default paths
        possible_paths = [
            "EcoCrop_DB_secondtrim.csv",
            "EcoCrop_DB.csv",
            os.path.join(os.path.dirname(__file__), "EcoCrop_DB_secondtrim.csv"),
            os.path.join(os.path.dirname(__file__), "EcoCrop_DB.csv"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break

    if csv_path is None or not os.path.exists(csv_path):
        raise FileNotFoundError("EcoCrop database CSV not found. Please provide the path.")

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} crops from EcoCrop database")
    return df


def get_crop_parameters(db: pd.DataFrame, crop_name: str) -> Dict:
    """
    Get parameters for a specific crop from the database.

    Args:
        db: EcoCrop database DataFrame
        crop_name: Scientific name or common name of the crop

    Returns:
        Dictionary with crop parameters
    """
    # Search by scientific name first
    mask = db['ScientificName'].str.lower() == crop_name.lower()

    # If not found, search in common names
    if not mask.any():
        mask = db['COMNAME'].str.lower().str.contains(crop_name.lower(), na=False)

    if not mask.any():
        available = db['ScientificName'].head(20).tolist()
        raise ValueError(f"Crop '{crop_name}' not found. Examples: {available}")

    row = db[mask].iloc[0]

    params = {
        'name': row['ScientificName'],
        'common_name': row.get('COMNAME', ''),
        # Temperature parameters (°C)
        'TOPMN': row.get('TOPMN', np.nan),  # Optimal temp min
        'TOPMX': row.get('TOPMX', np.nan),  # Optimal temp max
        'TMIN': row.get('TMIN', np.nan),    # Absolute temp min
        'TMAX': row.get('TMAX', np.nan),    # Absolute temp max
        'KTMP': row.get('KTMP', np.nan),    # Killing temperature
        # Precipitation parameters (mm/year)
        'ROPMN': row.get('ROPMN', np.nan),  # Optimal precip min
        'ROPMX': row.get('ROPMX', np.nan),  # Optimal precip max
        'RMIN': row.get('RMIN', np.nan),    # Absolute precip min
        'RMAX': row.get('RMAX', np.nan),    # Absolute precip max
        # Growing season (days)
        'GMIN': row.get('GMIN', np.nan),    # Min growing days
        'GMAX': row.get('GMAX', np.nan),    # Max growing days
        # Soil texture
        'TEXT': row.get('TEXT', ''),
        'TEXTR': row.get('TEXTR', ''),
        # Life form
        'LIFO': row.get('LIFO', ''),
        'HABI': row.get('HABI', ''),
    }

    print(f"✓ Loaded parameters for: {params['name']}")
    print(f"  Common names: {params['common_name'][:80]}...")
    print(f"  Optimal Temp: {params['TOPMN']}°C - {params['TOPMX']}°C")
    print(f"  Temp Range: {params['TMIN']}°C - {params['TMAX']}°C")
    print(f"  Optimal Precip: {params['ROPMN']}mm - {params['ROPMX']}mm/year")
    print(f"  Growing Season: {params['GMIN']} - {params['GMAX']} days")

    return params


def create_custom_crop_parameters(
    name: str,
    topmn: float, topmx: float,
    tmin: float, tmax: float,
    ropmn: float, ropmx: float,
    rmin: float, rmax: float,
    gmin: int = 90, gmax: int = 180,
    ktmp: float = None
) -> Dict:
    """
    Create custom crop parameters manually.

    Args:
        name: Crop name
        topmn/topmx: Optimal temperature range (°C)
        tmin/tmax: Absolute temperature range (°C)
        ropmn/ropmx: Optimal precipitation range (mm/year)
        rmin/rmax: Absolute precipitation range (mm/year)
        gmin/gmax: Growing season length (days)
        ktmp: Killing temperature (°C)

    Returns:
        Dictionary with crop parameters
    """
    return {
        'name': name,
        'common_name': name,
        'TOPMN': topmn,
        'TOPMX': topmx,
        'TMIN': tmin,
        'TMAX': tmax,
        'KTMP': ktmp if ktmp is not None else tmin - 5,
        'ROPMN': ropmn,
        'ROPMX': ropmx,
        'RMIN': rmin,
        'RMAX': rmax,
        'GMIN': gmin,
        'GMAX': gmax,
        'TEXT': '',
        'TEXTR': '',
    }


# =============================================================================
# Region of Interest (ROI) Functions
# =============================================================================

def create_roi_from_coordinates(
    lon: float, lat: float,
    buffer_km: float = 50
) -> ee.Geometry:
    """
    Create a region of interest from center coordinates.

    Args:
        lon: Longitude of center point
        lat: Latitude of center point
        buffer_km: Buffer radius in kilometers

    Returns:
        ee.Geometry object
    """
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(buffer_km * 1000)  # Convert km to meters
    print(f"✓ Created ROI: center ({lat}, {lon}), radius {buffer_km}km")
    return roi


def create_roi_from_bbox(
    min_lon: float, min_lat: float,
    max_lon: float, max_lat: float
) -> ee.Geometry:
    """
    Create a region of interest from bounding box.

    Args:
        min_lon, min_lat: Southwest corner
        max_lon, max_lat: Northeast corner

    Returns:
        ee.Geometry object
    """
    roi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    print(f"✓ Created ROI: bbox [{min_lon}, {min_lat}, {max_lon}, {max_lat}]")
    return roi


def create_roi_from_country(country_name: str) -> ee.Geometry:
    """
    Create a region of interest from country name.

    Args:
        country_name: Name of the country

    Returns:
        ee.Geometry object
    """
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    country = countries.filter(ee.Filter.eq('country_na', country_name))
    roi = country.geometry()
    print(f"✓ Created ROI for country: {country_name}")
    return roi


# =============================================================================
# Climate Data Fetching from GEE
# =============================================================================

def fetch_era5_climate_data(
    roi: ee.Geometry,
    start_date: str,
    end_date: str,
    scale: int = 11132  # ~0.1 degree
) -> ee.Image:
    """
    Fetch ERA5 climate data from GEE.

    Args:
        roi: Region of interest
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        scale: Resolution in meters

    Returns:
        ee.Image with mean temperature and total precipitation
    """
    # ERA5-Land Monthly data (available in GEE)
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi)

    # Calculate mean temperature (Kelvin to Celsius)
    mean_temp = era5.select('temperature_2m').mean() \
        .subtract(273.15).rename('mean_temp')

    # Calculate total precipitation (m to mm, monthly sum to annual)
    n_months = era5.size().getInfo()
    total_precip = era5.select('total_precipitation_sum').sum() \
        .multiply(1000).rename('total_precip')  # Convert m to mm

    # Min and Max temperature
    min_temp = era5.select('temperature_2m').min() \
        .subtract(273.15).rename('min_temp')
    max_temp = era5.select('temperature_2m').max() \
        .subtract(273.15).rename('max_temp')

    result = mean_temp.addBands(min_temp).addBands(max_temp).addBands(total_precip)

    print(f"✓ Fetched ERA5 climate data: {start_date} to {end_date}")
    return result


def fetch_terraclimate_data(
    roi: ee.Geometry,
    start_date: str,
    end_date: str
) -> ee.Image:
    """
    Fetch TerraClimate data (higher resolution alternative).

    Args:
        roi: Region of interest
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        ee.Image with climate variables
    """
    terraclimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi)

    # Temperature (already in Celsius * 10, so divide by 10)
    mean_temp = terraclimate.select('tmmx').mean().add(
        terraclimate.select('tmmn').mean()
    ).divide(20).rename('mean_temp')  # Average of max and min, divided by 10

    min_temp = terraclimate.select('tmmn').min().divide(10).rename('min_temp')
    max_temp = terraclimate.select('tmmx').max().divide(10).rename('max_temp')

    # Precipitation (mm)
    total_precip = terraclimate.select('pr').sum().rename('total_precip')

    result = mean_temp.addBands(min_temp).addBands(max_temp).addBands(total_precip)

    print(f"✓ Fetched TerraClimate data: {start_date} to {end_date}")
    return result


def fetch_worldclim_data(roi: ee.Geometry) -> ee.Image:
    """
    Fetch WorldClim BIO variables (1970-2000 climatology).

    Args:
        roi: Region of interest

    Returns:
        ee.Image with bioclimatic variables
    """
    worldclim = ee.Image("WORLDCLIM/V1/BIO")

    # BIO1 = Annual Mean Temperature (°C * 10)
    mean_temp = worldclim.select('bio01').divide(10).rename('mean_temp')

    # BIO6 = Min Temperature of Coldest Month (°C * 10)
    min_temp = worldclim.select('bio06').divide(10).rename('min_temp')

    # BIO5 = Max Temperature of Warmest Month (°C * 10)
    max_temp = worldclim.select('bio05').divide(10).rename('max_temp')

    # BIO12 = Annual Precipitation (mm)
    total_precip = worldclim.select('bio12').rename('total_precip')

    result = mean_temp.addBands(min_temp).addBands(max_temp).addBands(total_precip)

    print("✓ Fetched WorldClim climatology data (1970-2000)")
    return result


# =============================================================================
# Suitability Score Calculation
# =============================================================================

def calculate_temperature_score_gee(
    climate_image: ee.Image,
    crop_params: Dict,
    method: str = 'annual'
) -> ee.Image:
    """
    Calculate temperature suitability score using GEE.

    Args:
        climate_image: Image with temperature bands
        crop_params: Crop parameters dictionary
        method: 'annual' or 'perennial'

    Returns:
        ee.Image with temperature score (0-100)
    """
    topmn = crop_params['TOPMN']
    topmx = crop_params['TOPMX']
    tmin = crop_params['TMIN']
    tmax = crop_params['TMAX']

    mean_temp = climate_image.select('mean_temp')
    min_temp = climate_image.select('min_temp')
    max_temp = climate_image.select('max_temp')

    if method == 'perennial':
        # Linear scoring based on mean temperature
        # Score = 100 in optimal range, decreasing linearly outside

        # Below optimal
        score_below = mean_temp.subtract(tmin).divide(topmn - tmin).multiply(100)
        # In optimal range
        score_optimal = ee.Image.constant(100)
        # Above optimal
        score_above = ee.Image.constant(tmax).subtract(mean_temp) \
            .divide(tmax - topmx).multiply(100)

        score = ee.Image.constant(0) \
            .where(mean_temp.gte(tmin).And(mean_temp.lt(topmn)), score_below) \
            .where(mean_temp.gte(topmn).And(mean_temp.lte(topmx)), score_optimal) \
            .where(mean_temp.gt(topmx).And(mean_temp.lte(tmax)), score_above) \
            .where(mean_temp.lt(tmin).Or(mean_temp.gt(tmax)), 0)
    else:
        # Annual method: consider the range of temperatures
        # Check if the temperature range overlaps with suitable range

        # Simplified: use mean temperature for basic assessment
        score_below = mean_temp.subtract(tmin).divide(topmn - tmin).multiply(100)
        score_optimal = ee.Image.constant(100)
        score_above = ee.Image.constant(tmax).subtract(mean_temp) \
            .divide(tmax - topmx).multiply(100)

        score = ee.Image.constant(0) \
            .where(mean_temp.gte(tmin).And(mean_temp.lt(topmn)), score_below) \
            .where(mean_temp.gte(topmn).And(mean_temp.lte(topmx)), score_optimal) \
            .where(mean_temp.gt(topmx).And(mean_temp.lte(tmax)), score_above) \
            .where(mean_temp.lt(tmin).Or(mean_temp.gt(tmax)), 0)

        # Penalize if min/max temperatures exceed absolute limits
        penalty_cold = min_temp.lt(tmin).multiply(20)
        penalty_hot = max_temp.gt(tmax).multiply(20)
        score = score.subtract(penalty_cold).subtract(penalty_hot)

    # Clamp to 0-100
    score = score.clamp(0, 100).rename('temp_score')

    return score


def calculate_precipitation_score_gee(
    climate_image: ee.Image,
    crop_params: Dict
) -> ee.Image:
    """
    Calculate precipitation suitability score using GEE.

    Args:
        climate_image: Image with precipitation band
        crop_params: Crop parameters dictionary

    Returns:
        ee.Image with precipitation score (0-100)
    """
    ropmn = crop_params['ROPMN']
    ropmx = crop_params['ROPMX']
    rmin = crop_params['RMIN']
    rmax = crop_params['RMAX']

    precip = climate_image.select('total_precip')

    # Below optimal
    score_below = precip.subtract(rmin).divide(ropmn - rmin).multiply(100)
    # In optimal range
    score_optimal = ee.Image.constant(100)
    # Above optimal
    score_above = ee.Image.constant(rmax).subtract(precip) \
        .divide(rmax - ropmx).multiply(100)

    score = ee.Image.constant(0) \
        .where(precip.gte(rmin).And(precip.lt(ropmn)), score_below) \
        .where(precip.gte(ropmn).And(precip.lte(ropmx)), score_optimal) \
        .where(precip.gt(ropmx).And(precip.lte(rmax)), score_above) \
        .where(precip.lt(rmin).Or(precip.gt(rmax)), 0)

    # Clamp to 0-100
    score = score.clamp(0, 100).rename('precip_score')

    return score


def calculate_overall_suitability_gee(
    climate_image: ee.Image,
    crop_params: Dict,
    method: str = 'annual'
) -> ee.Image:
    """
    Calculate overall crop suitability score.

    The overall score is the minimum of temperature and precipitation scores
    (limiting factor approach).

    Args:
        climate_image: Image with climate bands
        crop_params: Crop parameters dictionary
        method: 'annual' or 'perennial'

    Returns:
        ee.Image with overall, temperature, and precipitation scores
    """
    temp_score = calculate_temperature_score_gee(climate_image, crop_params, method)
    precip_score = calculate_precipitation_score_gee(climate_image, crop_params)

    # Overall score is the minimum (limiting factor)
    overall_score = temp_score.min(precip_score).rename('overall_score')

    result = overall_score.addBands(temp_score).addBands(precip_score)

    return result


# =============================================================================
# Soil Suitability (Optional)
# =============================================================================

def fetch_soil_texture_gee(roi: ee.Geometry) -> ee.Image:
    """
    Fetch soil texture data from OpenLandMap.

    Args:
        roi: Region of interest

    Returns:
        ee.Image with soil texture class
    """
    # OpenLandMap Soil Texture Class
    soil = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")

    # Select topsoil (0cm depth)
    soil_texture = soil.select('b0').rename('soil_texture')

    print("✓ Fetched soil texture data from OpenLandMap")
    return soil_texture


def calculate_soil_suitability_gee(
    soil_image: ee.Image,
    crop_params: Dict
) -> ee.Image:
    """
    Calculate soil texture suitability.

    Soil texture classes (USDA):
    1: Clay, 2: Silty Clay, 3: Sandy Clay, 4: Clay Loam,
    5: Silty Clay Loam, 6: Sandy Clay Loam, 7: Loam,
    8: Silt Loam, 9: Sandy Loam, 10: Silt, 11: Loamy Sand, 12: Sand

    Args:
        soil_image: Image with soil texture
        crop_params: Crop parameters dictionary

    Returns:
        ee.Image with soil suitability score (0-100)
    """
    text = str(crop_params.get('TEXT', '')).lower()

    soil_texture = soil_image.select('soil_texture')

    # Default: all soils suitable
    score = ee.Image.constant(100)

    # Heavy soils: Clay, Silty Clay, Sandy Clay (1, 2, 3)
    heavy_mask = soil_texture.lte(3)

    # Medium soils: Clay Loam to Loam (4, 5, 6, 7, 8)
    medium_mask = soil_texture.gte(4).And(soil_texture.lte(8))

    # Light soils: Sandy Loam to Sand (9, 10, 11, 12)
    light_mask = soil_texture.gte(9)

    if 'heavy' in text and 'medium' not in text and 'light' not in text:
        score = score.where(medium_mask.Or(light_mask), 50)
    elif 'medium' in text and 'heavy' not in text and 'light' not in text:
        score = score.where(heavy_mask.Or(light_mask), 50)
    elif 'light' in text and 'heavy' not in text and 'medium' not in text:
        score = score.where(heavy_mask.Or(medium_mask), 50)

    return score.rename('soil_score')


# =============================================================================
# Result Analysis and Export
# =============================================================================

def get_regional_statistics(
    score_image: ee.Image,
    roi: ee.Geometry,
    scale: int = 1000
) -> Dict:
    """
    Calculate regional statistics for suitability scores.

    Args:
        score_image: Image with suitability scores
        roi: Region of interest
        scale: Resolution in meters

    Returns:
        Dictionary with statistics
    """
    stats = score_image.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), '', True
        ).combine(
            ee.Reducer.stdDev(), '', True
        ),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).getInfo()

    return stats


def export_to_drive(
    image: ee.Image,
    roi: ee.Geometry,
    description: str,
    folder: str = 'EcoCrop_GEE',
    scale: int = 1000
) -> ee.batch.Task:
    """
    Export image to Google Drive.

    Args:
        image: Image to export
        roi: Region of interest
        description: File name
        folder: Drive folder name
        scale: Resolution in meters

    Returns:
        Export task
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=roi,
        scale=scale,
        crs='EPSG:4326',
        maxPixels=1e9
    )
    task.start()
    print(f"✓ Export task started: {description}")
    print(f"  Check progress at: https://code.earthengine.google.com/tasks")
    return task


# =============================================================================
# Visualization
# =============================================================================

def get_visualization_params() -> Dict:
    """Get visualization parameters for suitability maps."""
    return {
        'suitability': {
            'min': 0,
            'max': 100,
            'palette': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
        },
        'temperature': {
            'min': -10,
            'max': 40,
            'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#fee090', '#fdae61', '#f46d43', '#d73027']
        },
        'precipitation': {
            'min': 0,
            'max': 3000,
            'palette': ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        }
    }


def create_folium_map(
    score_image: ee.Image,
    roi: ee.Geometry,
    center_lat: float,
    center_lon: float,
    crop_name: str
):
    """
    Create an interactive Folium map with suitability scores.

    Requires: pip install folium geemap

    Args:
        score_image: Image with suitability scores
        roi: Region of interest
        center_lat, center_lon: Map center
        crop_name: Name of the crop

    Returns:
        Folium map object
    """
    try:
        import folium
        import geemap.foliumap as geemap
    except ImportError:
        print("Please install folium and geemap: pip install folium geemap")
        return None

    # Create map
    m = geemap.Map(center=[center_lat, center_lon], zoom=8)

    vis_params = get_visualization_params()

    # Add layers
    m.addLayer(
        score_image.select('overall_score'),
        vis_params['suitability'],
        f'{crop_name} - Overall Suitability'
    )
    m.addLayer(
        score_image.select('temp_score'),
        vis_params['suitability'],
        f'{crop_name} - Temperature Score',
        shown=False
    )
    m.addLayer(
        score_image.select('precip_score'),
        vis_params['suitability'],
        f'{crop_name} - Precipitation Score',
        shown=False
    )

    # Add ROI boundary
    m.addLayer(ee.Image().paint(roi, 0, 2), {'palette': 'red'}, 'ROI Boundary')

    # Add legend
    m.add_colorbar(
        vis_params['suitability'],
        label='Suitability Score (0-100)',
        layer_name=f'{crop_name} - Overall Suitability'
    )

    return m


# =============================================================================
# Main Analysis Class
# =============================================================================

class EcoCropGEE:
    """
    Main class for EcoCrop suitability analysis using Google Earth Engine.
    """

    def __init__(self, project: str = None):
        """
        Initialize EcoCrop GEE analyzer.

        Args:
            project: GEE project ID (optional)
        """
        initialize_gee(project)
        self.db = None
        self.crop_params = None
        self.roi = None
        self.climate_data = None
        self.scores = None

    def load_database(self, csv_path: str = None):
        """Load EcoCrop database."""
        self.db = load_ecocrop_database(csv_path)
        return self

    def set_crop(self, crop_name: str = None, custom_params: Dict = None):
        """
        Set crop for analysis.

        Args:
            crop_name: Name of crop from database
            custom_params: Custom crop parameters (alternative to crop_name)
        """
        if custom_params:
            self.crop_params = custom_params
        elif crop_name:
            if self.db is None:
                self.load_database()
            self.crop_params = get_crop_parameters(self.db, crop_name)
        else:
            raise ValueError("Provide either crop_name or custom_params")
        return self

    def set_roi_point(self, lon: float, lat: float, buffer_km: float = 50):
        """Set ROI from center point."""
        self.roi = create_roi_from_coordinates(lon, lat, buffer_km)
        self.center_lon = lon
        self.center_lat = lat
        return self

    def set_roi_bbox(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float):
        """Set ROI from bounding box."""
        self.roi = create_roi_from_bbox(min_lon, min_lat, max_lon, max_lat)
        self.center_lon = (min_lon + max_lon) / 2
        self.center_lat = (min_lat + max_lat) / 2
        return self

    def set_roi_country(self, country_name: str):
        """Set ROI from country name."""
        self.roi = create_roi_from_country(country_name)
        # Get centroid
        centroid = self.roi.centroid().coordinates().getInfo()
        self.center_lon, self.center_lat = centroid
        return self

    def fetch_climate(
        self,
        start_date: str = None,
        end_date: str = None,
        source: str = 'era5'
    ):
        """
        Fetch climate data.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: 'era5', 'terraclimate', or 'worldclim'
        """
        if self.roi is None:
            raise ValueError("Set ROI first using set_roi_* methods")

        if source == 'worldclim':
            self.climate_data = fetch_worldclim_data(self.roi)
        elif source == 'terraclimate':
            if not start_date or not end_date:
                raise ValueError("start_date and end_date required for TerraClimate")
            self.climate_data = fetch_terraclimate_data(self.roi, start_date, end_date)
        else:  # era5
            if not start_date or not end_date:
                raise ValueError("start_date and end_date required for ERA5")
            self.climate_data = fetch_era5_climate_data(self.roi, start_date, end_date)

        return self

    def calculate_suitability(self, method: str = 'annual'):
        """
        Calculate crop suitability scores.

        Args:
            method: 'annual' or 'perennial'
        """
        if self.climate_data is None:
            raise ValueError("Fetch climate data first")
        if self.crop_params is None:
            raise ValueError("Set crop first")

        self.scores = calculate_overall_suitability_gee(
            self.climate_data,
            self.crop_params,
            method
        )

        print(f"✓ Calculated suitability scores for {self.crop_params['name']}")
        return self

    def get_statistics(self, scale: int = 1000) -> Dict:
        """Get regional statistics."""
        if self.scores is None:
            raise ValueError("Calculate suitability first")

        stats = get_regional_statistics(self.scores, self.roi, scale)

        print("\n" + "=" * 50)
        print(f"Regional Statistics for {self.crop_params['name']}")
        print("=" * 50)

        for key, value in stats.items():
            if value is not None:
                print(f"  {key}: {value:.2f}")

        return stats

    def create_map(self):
        """Create interactive map."""
        if self.scores is None:
            raise ValueError("Calculate suitability first")

        return create_folium_map(
            self.scores,
            self.roi,
            self.center_lat,
            self.center_lon,
            self.crop_params['name']
        )

    def export(self, filename: str, folder: str = 'EcoCrop_GEE', scale: int = 1000):
        """Export results to Google Drive."""
        if self.scores is None:
            raise ValueError("Calculate suitability first")

        return export_to_drive(self.scores, self.roi, filename, folder, scale)


# =============================================================================
# Example Usage
# =============================================================================

def example_analysis():
    """
    Example: Analyze wheat suitability in South Korea.
    """
    print("\n" + "=" * 60)
    print("EcoCrop GEE Analysis Example")
    print("=" * 60 + "\n")

    # Initialize analyzer
    analyzer = EcoCropGEE()

    # Load database and set crop
    analyzer.load_database()
    analyzer.set_crop('wheat')  # or 'Triticum aestivum'

    # Set region of interest (South Korea example)
    # Option 1: Point with buffer
    analyzer.set_roi_point(lon=127.0, lat=37.5, buffer_km=100)

    # Option 2: Bounding box
    # analyzer.set_roi_bbox(125.0, 33.0, 130.0, 38.5)

    # Option 3: Country
    # analyzer.set_roi_country('South Korea')

    # Fetch climate data
    # Option 1: WorldClim (climatology 1970-2000, no date needed)
    analyzer.fetch_climate(source='worldclim')

    # Option 2: Recent data from ERA5
    # analyzer.fetch_climate(start_date='2020-01-01', end_date='2020-12-31', source='era5')

    # Option 3: TerraClimate
    # analyzer.fetch_climate(start_date='2020-01-01', end_date='2020-12-31', source='terraclimate')

    # Calculate suitability
    analyzer.calculate_suitability(method='annual')

    # Get statistics
    stats = analyzer.get_statistics()

    # Create interactive map (requires folium, geemap)
    try:
        m = analyzer.create_map()
        if m:
            m.save('wheat_suitability_map.html')
            print("\n✓ Map saved to wheat_suitability_map.html")
    except Exception as e:
        print(f"\nCould not create map: {e}")

    # Export to Drive (uncomment to use)
    # analyzer.export('wheat_suitability_korea', folder='EcoCrop_GEE')

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    return analyzer


def example_custom_crop():
    """
    Example: Analyze with custom crop parameters.
    """
    print("\n" + "=" * 60)
    print("Custom Crop Analysis Example")
    print("=" * 60 + "\n")

    # Create custom crop parameters
    custom_params = create_custom_crop_parameters(
        name="Custom Rice Variety",
        topmn=22, topmx=30,   # Optimal temp: 22-30°C
        tmin=15, tmax=38,     # Absolute temp: 15-38°C
        ropmn=1000, ropmx=2000,  # Optimal precip: 1000-2000mm
        rmin=600, rmax=3000,     # Absolute precip: 600-3000mm
        gmin=120, gmax=180,      # Growing season: 120-180 days
        ktmp=10                  # Killing temperature: 10°C
    )

    # Initialize and run analysis
    analyzer = EcoCropGEE()
    analyzer.set_crop(custom_params=custom_params)
    analyzer.set_roi_point(lon=127.0, lat=35.0, buffer_km=50)
    analyzer.fetch_climate(source='worldclim')
    analyzer.calculate_suitability(method='annual')

    stats = analyzer.get_statistics()

    return analyzer


if __name__ == "__main__":
    # Run example analysis
    # Uncomment the example you want to run:

    # Example 1: Database crop
    analyzer = example_analysis()

    # Example 2: Custom crop
    # analyzer = example_custom_crop()
