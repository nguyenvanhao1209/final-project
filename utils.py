import base64
import json

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