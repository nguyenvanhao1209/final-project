import base64
import json
from datetime import datetime

def get_name_email(token):
    # Split the token into its three parts: header, payload, and signature
    parts = token.split('.')
    
    # Decode each part using Base64 decoding
    decoded_parts = []
    for part in parts:
        # Base64 urlsafe decoding with padding
        decoded_parts.append(base64.urlsafe_b64decode(part + '=' * (4 - len(part) % 4)))
    
    # Interpret the decoded parts as JSON
    decoded_header = json.loads(decoded_parts[0])
    decoded_payload = json.loads(decoded_parts[1])
    decoded_signature = decoded_parts[2]

    if 'email' in decoded_payload:
        email = decoded_payload['email']
    else:
        email = None

    if 'name' in decoded_payload:
        name = decoded_payload['name']
    else:
        name = None
    
    return name, email

def time_difference(date_string):
    # Parse the input date string
    input_date = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    
    # Get the current date and time
    current_date = datetime.now()
    
    # Calculate the time difference
    time_delta = current_date - input_date
    
    # Extract the components of the time difference
    days = time_delta.days
    seconds = time_delta.seconds
    
    # Check if the difference is in years
    if days > 365:
        years = days // 365
        if years == 1:
            return "1 year ago"
        else:
            return f"{years} years ago"
    
    # Check if the difference is in months
    elif days > 30:
        months = days // 30
        if months == 1:
            return "1 month ago"
        else:
            return f"{months} months ago"
    
    # Check if the difference is in days
    elif days > 0:
        if days == 1:
            return "1 day ago"
        else:
            return f"{days} days ago"
    
    # Check if the difference is in hours
    elif seconds // 3600 > 0:
        hours = seconds // 3600
        if hours == 1:
            return "1 hour ago"
        else:
            return f"{hours} hours ago"
    
    # Otherwise, the difference is in minutes
    else:
        minutes = seconds // 60
        if minutes == 1:
            return "1 minute ago"
        else:
            return f"{minutes} minutes ago"
        
def format_file_size(size):
    size = int(size)
    suffixes = ['B', 'kB', 'MB', 'GB', 'TB']  # List of size suffixes
    suffix_index = 0
    while size >= 1024 and suffix_index < len(suffixes) - 1:
        size /= 1024
        suffix_index += 1
    return f"{size:.2f}{suffixes[suffix_index]}"

def get_file_extension(url):
    # Split the URL by '/'
    file_name = url.split("/")[-1]

    # Split the file name by '.'
    file_extension = file_name.split(".")[-1]

    return file_extension

def get_file(url):
        # Split the URL by '/'
    file_name = url.split("/")[-1]

    return file_name

def calculate_file_size(number, size_type):
    size_types = ['B', 'kB', 'MB', 'GB', 'TB']
    if size_type not in size_types:
        raise ValueError(f"Invalid size type. Expected one of: {size_types}")

    size_index = size_types.index(size_type)
    size_in_bits = number * (1024 ** size_index)

    return size_in_bits

