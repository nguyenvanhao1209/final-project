import base64
import json
from datetime import datetime
import streamlit as st
from streamlit_star_rating import st_star_rating
import requests
from PIL import Image
from io import BytesIO
import streamlit.components.v1 as components

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

def display_star_rating(score):
    # Function to generate star ratings with Tailwind CSS
    def generate_star_html(score):
        tailwind_css = '''
        <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/tailwindcss/2.2.19/tailwind.min.css');
        .star {
            color: #fbbf24; /* Tailwind amber-400 color */
            font-size: 2rem;
            display: inline-block;
            position: relative;
        }
        .star::before {
            content: '★';
            color: #e5e7eb; /* Tailwind gray-300 color for empty star */
            position: absolute;
            left: 0;
            top: 0;
            z-index: 0;
        }
        .star.full::before {
            color: #fbbf24; /* Tailwind amber-400 color for full star */
            z-index: 1;
        }
        .star.half::before {
            background: linear-gradient(90deg, #fbbf24 50%, #e5e7eb 50%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            z-index: 1;
        }
        </style>
        '''

        stars_html = '<div class="flex">'
        full_stars = int(score)
        half_star = (score - full_stars) >= 0.5
        empty_stars = 5 - full_stars - int(half_star)

        stars_html += ''.join('<div class="star full">★</div>' for _ in range(full_stars))
        if half_star:
            stars_html += '<div class="star half">★</div>'
        stars_html += ''.join('<div class="star">★</div>' for _ in range(empty_stars))

        stars_html += '</div>'

        return tailwind_css + stars_html

    # Generate the star HTML
    star_html = generate_star_html(score)

    # Display the stars using components
    components.html(star_html, height=40)

def display_vote_detail(values, point, total):
    # Inject Tailwind CSS
    st.markdown("""
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        """, unsafe_allow_html=True)

    # Create a custom progress bar with Tailwind CSS matching the uploaded image
    def custom_progress_bar(label, value, max_value=total):
        st.markdown(f"""
        <div class="flex items-center mb-4">
            <div class="w-16 text-right mr-4 text-xs">{label}</div>
            <div class="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                <div class="bg-yellow-300 h-full rounded-full" style="width: {value/max_value*100}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,3])
    with col1:
        st.markdown(f"""
            <div class="text-left text-8xl" style="margin-top: -10px;">
                {point}
            </div>
            """, unsafe_allow_html=True)
        display_star_rating(point)
        st.markdown(f"""
            <div class="text-justify text-base font-semibold" style="margin-top: -10px;">
                {total} reviews
            </div>
            """, unsafe_allow_html=True)
    with col2:
        for i, value in enumerate(values, 1):
            custom_progress_bar(str(6 - i), value)

def image_with_name(name, size):
    param = {
        "name": name,
        "rounded": True,
        "size": size
    }

    url = requests.Request('GET', 'https://ui-avatars.com/api/', params=param).prepare().url

    return url
