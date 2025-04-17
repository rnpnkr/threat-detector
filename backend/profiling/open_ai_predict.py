import os
import json
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Define the profiling function schema for OpenAI function calling
def get_profiling_schema():
    return {
        'name': 'extract_threat_profiling_data',
        'description': 'Extract profiling data from CCTV footage for potential threat identification, focusing on concise descriptions for suspect identification as per police requirements.',
        'parameters': {
            'type': 'object',
            'properties': {
                'facial_features': {
                    'type': 'object',
                    'description': 'Details of facial features if visible, for suspect identification.',
                    'properties': {
                        'approximate_age': {'type': 'string', 'description': 'Approximate age (e.g., 20s, 30s)'},
                        'gender': {'type': 'string', 'description': 'Perceived gender (e.g., male, female)'},
                        'skin_tone': {'type': 'string', 'description': 'Brief skin tone description (e.g., light, medium, dark)'},
                        'facial_hair': {'type': 'string', 'description': 'Brief note on facial hair if present (e.g., beard, mustache, none)'},
                        'distinguishing_marks': {'type': 'string', 'description': 'Visible distinguishing marks (e.g., scars, tattoos, none)'}
                    },
                    'required': ['approximate_age', 'gender', 'skin_tone', 'facial_hair', 'distinguishing_marks']
                },
                'clothing_accessories': {
                    'type': 'object',
                    'description': 'Details of clothing and accessories to aid in identifying the suspect in other footage.',
                    'properties': {
                        'attire_color': {'type': 'string', 'description': 'Primary color of attire (e.g., black, blue)'},
                        'attire_type': {'type': 'string', 'description': 'Type of attire (e.g., shirt, jacket, hoodie)'},
                        'logos_or_patterns': {'type': 'string', 'description': 'Visible logos or patterns on clothing (e.g., Nike logo, striped, none)'},
                        'hats': {'type': 'string', 'description': 'Type of hat if worn (e.g., baseball cap, beanie, none)'},
                        'glasses': {'type': 'string', 'description': 'Type of glasses if worn (e.g., sunglasses, prescription, none)'},
                        'bags': {'type': 'string', 'description': 'Type of bag if carried (e.g., backpack, messenger bag, none)'}
                    },
                    'required': ['attire_color', 'attire_type', 'logos_or_patterns', 'hats', 'glasses', 'bags']
                },
                'body_characteristics': {
                    'type': 'object',
                    'description': 'Physical characteristics to help estimate suspect profile from CCTV footage.',
                    'properties': {
                        'posture': {'type': 'string', 'description': 'Brief description of posture (e.g., upright, slouched)'},
                        'build': {'type': 'string', 'description': 'Body build (e.g., slim, average, muscular)'},
                        'relative_height': {'type': 'string', 'description': 'Estimated height range in feet (e.g., 5.5-6.0 ft)'}
                    },
                    'required': ['posture', 'build', 'relative_height']
                }
            },
            'required': ['facial_features', 'clothing_accessories', 'body_characteristics']
        }
    }

# Function to predict profiling data from image using OpenAI
def predict_profiling_data(image_path):
    try:
        # Encode the image
        base64_image = encode_image(image_path)

        # Create chat completion request with function calling
        chat_completion = client.chat.completions.create(
            model='gpt-4o',
            max_tokens=300,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Analyze this CCTV footage image for potential threat identification. Extract concise details about facial features (if visible), clothing/accessories, and body characteristics to aid in suspect identification as per police requirements.'
                        },
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/jpeg;base64,{base64_image}'
                            }
                        }
                    ]
                }
            ],
            functions=[get_profiling_schema()],
            function_call={'name': 'extract_threat_profiling_data'}
        )

        # Extract the function call arguments from the response
        response_message = chat_completion.choices[0].message
        profiling_data = json.loads(response_message.function_call.arguments)

        print('Profiling data extracted successfully.')
        return profiling_data

    except Exception as e:
        print(f'Error processing image with OpenAI: {str(e)}')
        return {
            'error': str(e),
            'facial_features': {},
            'clothing_accessories': {},
            'body_characteristics': {}
        } 