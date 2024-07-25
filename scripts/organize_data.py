import os
import re
import csv
from datetime import datetime, timedelta


# Helper function to parse dates from filenames
def parse_dates_from_filename(filename, pattern, date_format='%Y%m%d'):
    matches = re.findall(pattern, filename)
    if matches:
        date_strings = [date_str for match in matches for date_str in match if date_str]
        return [datetime.strptime(date_str, date_format) for date_str in date_strings]
    return []

# Helper function to parse fire ID from filenames
def parse_fire_id_from_filename(filename, pattern):
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

# Function to find matching files
def find_matching_files(lum_change_dir, binary_product_dir, coh_change_dir, dnbr_dir, date_tolerance=21):
    date_tolerance = timedelta(days=date_tolerance)
    
    lum_change_files = os.listdir(lum_change_dir)
    binary_product_files = os.listdir(binary_product_dir)
    coh_change_files = os.listdir(coh_change_dir)
    dnbr_files = os.listdir(dnbr_dir)
    
    # Regex patterns to extract dates and fire ID from filenames
    lum_change_pattern = r'_(\d{8})_(\d{8})_'  # Matches two dates in lum_change filenames
    binary_product_pattern = r'_(\d{8})_(\d{8})\.tif'  # Matches two dates in binary_product filenames
    coh_change_pattern = r'_(\d{8})_(\d{8})_(\d{8})'  # Matches three dates in coh_change filenames
    dnbr_pattern = r'_(\d{8}T\d{6})_(\d{8}T\d{6})_'  # Matches two dates in dnbr filenames
    lum_fire_id_pattern = r'^(\d{3,5})_'  # Matches fire ID at the start of the filename (3, 4, or 5 digits)
    dnbr_fire_id_pattern = r'_(\d{3,5})_\d{8}T\d{6}_\d{8}T\d{6}_10m_DNBR\.TIF'  # Matches fire ID before the first date in dnbr filenames (3, 4, or 5 digits)
    binary_product_fire_id_pattern = r'^(\d{3,5})_'  # Matches fire ID in binary_product filenames
    coh_change_fire_id_pattern = r'^(\d{3,5})_'  # Matches fire ID in coh_change filenames

    # Dictionary to store matches
    matches = []
    no_match_reasons = {}

    for lum_file in lum_change_files:
        lum_fire_id = parse_fire_id_from_filename(lum_file, lum_fire_id_pattern)
        lum_dates = parse_dates_from_filename(lum_file, lum_change_pattern, date_format='%Y%m%d')
        
        if len(lum_dates) < 2 or not lum_fire_id:
            no_match_reasons[lum_file] = "Couldn't extract fire ID or not enough dates"
            continue  # Skip files that do not match the expected pattern
        
        # Finding the closest match for dNBR files
        closest_dnbr_file = None
        closest_dnbr_date_diff = timedelta(days=9999)
        
        for dnbr_file in dnbr_files:
            if not dnbr_file.startswith('S2B'):
                continue  # Skipping dNBR files not starting with 'S2B'

            dnbr_fire_id = parse_fire_id_from_filename(dnbr_file, dnbr_fire_id_pattern)
            dnbr_dates = parse_dates_from_filename(dnbr_file, dnbr_pattern, date_format='%Y%m%dT%H%M%S')
            
            if dnbr_fire_id != lum_fire_id:
                continue  # Skipping if fire IDs do not match
            
            for dnbr_date in dnbr_dates:
                for lum_date in lum_dates:
                    if abs(dnbr_date - lum_date) < closest_dnbr_date_diff:
                        closest_dnbr_date_diff = abs(dnbr_date - lum_date)
                        closest_dnbr_file = dnbr_file

        if not closest_dnbr_file:
            no_match_reasons[lum_file] = "Couldn't find a matching dNBR file"
            closest_dnbr_file = "None"

        # Finding exact matches for binary_product and coh_change files
        binary_product_match = None
        coh_change_match = None
        
        for binary_file in binary_product_files:
            binary_fire_id = parse_fire_id_from_filename(binary_file, binary_product_fire_id_pattern)
            binary_dates = parse_dates_from_filename(binary_file, binary_product_pattern, date_format='%Y%m%d')
            
            if binary_fire_id == lum_fire_id and len(binary_dates) == 2:
                if all(lum_date in binary_dates for lum_date in lum_dates):
                    binary_product_match = binary_file
                    break
        
        if not binary_product_match:
            no_match_reasons[lum_file] = "Couldn't find a matching binary_product file"
            binary_product_match = "None"

        for coh_file in coh_change_files:
            coh_fire_id = parse_fire_id_from_filename(coh_file, coh_change_fire_id_pattern)
            coh_dates = parse_dates_from_filename(coh_file, coh_change_pattern, date_format='%Y%m%d')
            
            if coh_fire_id == lum_fire_id:
                if sum(1 for lum_date in lum_dates if any(coh_date.date() == lum_date.date() for coh_date in coh_dates)) >= 2:
                    coh_change_match = coh_file
                    break
        
        if not coh_change_match:
            no_match_reasons[lum_file] = "Couldn't find an exact matching coh_change file. Found closest match instead."
            closest_coh_file = "None"
        else:
            closest_coh_file = coh_change_match

        # Find closest files for unmatched cases

        if not binary_product_match or not coh_change_match:
            if not binary_product_match:
                binary_product_match = find_closest_file(lum_dates, binary_product_files, binary_product_fire_id_pattern, binary_product_pattern, lum_fire_id)
            if not coh_change_match:
                coh_change_match = find_closest_file(lum_dates, coh_change_files, coh_change_fire_id_pattern, coh_change_pattern, lum_fire_id)
            
        matches.append({
            'lum_change': os.path.join(lum_change_dir, lum_file),
            'coh_change': os.path.join(coh_change_dir, coh_change_match),
            'dnbr': os.path.join(dnbr_dir, closest_dnbr_file),
            'binary_product': os.path.join(binary_product_dir, binary_product_match)
        })
    
    return matches, no_match_reasons

def find_closest_file(lum_dates, file_list, fire_id_pattern, date_pattern, lum_fire_id):
    closest_file = None
    closest_date_diff = timedelta(days=9999)
    
    for file in file_list:
        fire_id = parse_fire_id_from_filename(file, fire_id_pattern)
        dates = parse_dates_from_filename(file, date_pattern, date_format='%Y%m%d' if 'T' not in date_pattern else '%Y%m%dT%H%M%S')
        
        if fire_id != lum_fire_id:
            continue
        
        for date in dates:
            for lum_date in lum_dates:
                if abs(date - lum_date) < closest_date_diff:
                    closest_date_diff = abs(date - lum_date)
                    closest_file = file
    
    return closest_file or "None"

# Set image directories
lum_change_dir = '/Bhaltos/ASHWATH/SAR_data_formatted_v2/lum_change'
binary_product_dir = '/Bhaltos/ASHWATH/SAR_data_formatted_v2/binary_products'
coh_change_dir = '/Bhaltos/ASHWATH/SAR_data_formatted_v2/coh_change'
dnbr_dir = '/Bhaltos/ASHWATH/Dataset/dNBR'
csv_file_path = '/Bhaltos/ASHWATH/metadata.csv'

# Find matching files
matches, no_match_reasons = find_matching_files(lum_change_dir, binary_product_dir, coh_change_dir, dnbr_dir)

# Write matches to CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['lum_change', 'coh_change', 'dnbr', 'binary_product'])
    writer.writeheader()
    for match in matches:
        writer.writerow(match)

# Print reasons for no matches, and alternative action taken, if any
for lum_file, reason in no_match_reasons.items():
    print(f"Lum Change File: {lum_file} - Reason: {reason}")