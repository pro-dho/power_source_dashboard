"""Data loading module for Light Source Power Dashboard."""
import os
import pandas as pd
import yaml
from io import StringIO
from typing import Dict, Any, List, Optional


def parse_multi_section_csv(filepath: str) -> Dict[str, Any]:
    """Parse multi-section CSV file with # headers.
    
    Returns a dict with:
    - 'light_sources': DataFrame
    - 'power_meters': DataFrame
    - 'acquisition_datetime': str
    - 'power_measurements': DataFrame
    """
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    sections = {}
    current_section = None
    current_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check for section header (# or ##)
        if stripped.startswith('#'):
            # Save previous section if exists
            if current_section is not None and current_lines:
                sections[current_section] = current_lines
            
            # Get new section name
            section_name = stripped.lstrip('#').strip()
            
            # Skip meta-sections like "input_data" that have no data
            if section_name in ['input_data']:
                continue
                
            current_section = section_name
            current_lines = []
        elif stripped:  # Non-empty line
            current_lines.append(line)
    
    # Save last section
    if current_section is not None and current_lines:
        sections[current_section] = current_lines
    
    # Convert sections to appropriate types
    result = {}
    for name, lines in sections.items():
        content = ''.join(lines)
        
        if name == 'acquisition_datetime':
            result[name] = content.strip().strip('"').replace(',', '.')
        elif content.strip():
            try:
                df = pd.read_csv(StringIO(content))
                if 'acquisition_datetime' in df.columns:
                    df['acquisition_datetime'] = (
                        df['acquisition_datetime']
                        .str.replace(',', '.')
                        .str.strip('"')
                    )
                    df['acquisition_datetime'] = pd.to_datetime(df['acquisition_datetime'])
                result[name] = df
            except Exception:
                result[name] = content
    
    return result


def load_key_measurements(filepath: str) -> pd.DataFrame:
    """Load key measurements from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return pd.DataFrame(data)


def load_input_parameters(filepath: str) -> Dict[str, Any]:
    """Load input parameters from YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_dataset(dataset_path: str) -> Dict[str, Any]:
    """Load all data from a dataset folder.
    
    Returns a dict with:
    - 'light_sources': DataFrame
    - 'power_meters': DataFrame
    - 'power_measurements': DataFrame
    - 'key_measurements': DataFrame
    - 'input_parameters': dict
    """
    csv_path = os.path.join(dataset_path, 'Model_LightSourcePower_v2.csv')
    key_path = os.path.join(dataset_path, 'dataset_key_measurements.yaml')
    params_path = os.path.join(dataset_path, 'dataset_input_parameters.yaml')
    
    # Parse CSV
    csv_data = parse_multi_section_csv(csv_path)
    
    # Load YAML files
    key_measurements = load_key_measurements(key_path)
    input_parameters = load_input_parameters(params_path)
    
    return {
        'light_sources': csv_data.get('light_sources', pd.DataFrame()),
        'power_meters': csv_data.get('power_meters', pd.DataFrame()),
        'power_measurements': csv_data.get('power_measurements', pd.DataFrame()),
        'key_measurements': key_measurements,
        'input_parameters': input_parameters,
    }


def load_all_datasets(base_path: str) -> Dict[str, Dict[str, Any]]:
    """Load all datasets from the base path.
    
    Returns a dict keyed by dataset folder name (e.g., '0', '1').
    """
    datasets = {}
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            try:
                datasets[folder] = load_dataset(folder_path)
            except Exception as e:
                print(f"Error loading dataset {folder}: {e}")
    
    return datasets


def get_filter_options(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract unique filter options from loaded data."""
    pm = data.get('power_measurements', pd.DataFrame())
    
    options = {
        'light_sources': [],
        'power_meters': [],
        'measuring_locations': [],
        'power_set_points': [],
    }
    
    if not pm.empty:
        if 'light_source' in pm.columns:
            options['light_sources'] = sorted(pm['light_source'].unique().tolist())
        if 'power_meter' in pm.columns:
            options['power_meters'] = sorted(pm['power_meter'].unique().tolist())
        if 'measuring_location' in pm.columns:
            options['measuring_locations'] = sorted(pm['measuring_location'].unique().tolist())
        if 'power_set_point' in pm.columns:
            options['power_set_points'] = sorted(pm['power_set_point'].unique().tolist())
    
    return options


def filter_measurements(
    df: pd.DataFrame,
    light_source: Optional[str] = None,
    power_meter: Optional[str] = None,
    measuring_location: Optional[str] = None,
    start_datetime: Optional[pd.Timestamp] = None,
    end_datetime: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Filter power measurements DataFrame based on criteria."""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    if light_source and 'light_source' in filtered.columns:
        filtered = filtered[filtered['light_source'] == light_source]
    
    if power_meter and 'power_meter' in filtered.columns:
        filtered = filtered[filtered['power_meter'] == power_meter]
    
    if measuring_location and 'measuring_location' in filtered.columns:
        filtered = filtered[filtered['measuring_location'] == measuring_location]
    
    if start_datetime and 'acquisition_datetime' in filtered.columns:
        filtered = filtered[filtered['acquisition_datetime'] >= start_datetime]
    
    if end_datetime and 'acquisition_datetime' in filtered.columns:
        filtered = filtered[filtered['acquisition_datetime'] <= end_datetime]
    
    return filtered


def get_linearity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get data for linearity plot (multiple power set points)."""
    if df.empty or 'power_set_point' not in df.columns:
        return pd.DataFrame()
    
    # Group by power_set_point and get mean power
    return df.groupby('power_set_point').agg({
        'power_mw': ['mean', 'std', 'count']
    }).reset_index()


def get_short_term_stability_data(
    df: pd.DataFrame,
    integration_time: float = 0.01
) -> pd.DataFrame:
    """Get data for short-term stability plot (same set point, short intervals)."""
    if df.empty or 'integration_time_seconds' not in df.columns:
        return pd.DataFrame()
    
    # Filter for short integration time measurements at same set point
    return df[df['integration_time_seconds'] == integration_time].copy()


def get_long_term_stability_data(
    df: pd.DataFrame,
    integration_time: float = 0.1
) -> pd.DataFrame:
    """Get data for long-term stability plot (same set point, longer intervals)."""
    if df.empty or 'integration_time_seconds' not in df.columns:
        return pd.DataFrame()
    
    # Filter for longer integration time measurements
    return df[df['integration_time_seconds'] == integration_time].copy()


def get_global_metrics(datasets: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Aggregate key measurements from all datasets into a single DataFrame.
    
    Adds a 'Dataset' column to identify the source.
    """
    all_metrics = []
    
    for dataset_name, data in datasets.items():
        key_meas = data.get('key_measurements', pd.DataFrame())
        if not key_meas.empty:
            # Add dataset name column
            df = key_meas.copy()
            df.insert(0, 'Dataset', dataset_name)
            
            # Ensure date columns are datetime objects
            date_cols = [c for c in df.columns if 'datetime' in c.lower()]
            for col in date_cols:
                # Handle potential mixed types or '0001-01-01' values
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
            all_metrics.append(df)
    
    if all_metrics:
        result = pd.concat(all_metrics, ignore_index=True)
        # Sort by date if available, otherwise by dataset
        if 'acquisition_datetime' in result.columns:
            result = result.sort_values('acquisition_datetime', ascending=False)
        elif 'power_linearity_start_datetime' in result.columns:
            result = result.sort_values('power_linearity_start_datetime', ascending=False)
        return result
        
    return pd.DataFrame()
