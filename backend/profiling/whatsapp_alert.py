import os
import json
from openai import OpenAI
from dotenv import load_dotenv
# Import the predict_profiling_data function from open_ai_predict.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from open_ai_predict import predict_profiling_data

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Remove hardcoded profiling data
# profiling_data = {...}

# Function to generate a concise profiling message in the specified language
def generate_profiling_message(profiling_data, language='English'):
    try:
        # Create chat completion request to generate a concise message
        chat_completion = client.chat.completions.create(
            model='gpt-4',
            max_tokens=100,
            messages=[
                {
                    'role': 'system',
                    'content': f'You are an assistant that generates concise profiling messages for potential threats identified in CCTV footage. The message should be in {language} and summarize key identifying features for quick suspect identification. Format the message with newlines for readability on WhatsApp.'
                },
                {
                    'role': 'user',
                    'content': f'Generate a concise profiling message in 30-35 words maximum for a potential threat based on this data: {json.dumps(profiling_data, indent=2)}'
                }
            ]
        )

        # Extract the generated message from the response
        message = chat_completion.choices[0].message.content.strip()
        print(f'Generated {language} message: {message}')
        return message
    except Exception as e:
        print(f'Error generating message with OpenAI: {str(e)}')
        return f'Error generating message: {str(e)}'

# Function to send the message and image to a WhatsApp number
def send_whatsapp_message(message, whatsapp_number, image_path=None):
    from twilio.rest import Client
    from twilio.base.exceptions import TwilioRestException
    
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    from_whatsapp_number = os.getenv('TWILIO_WHATSAPP_NUMBER')
    ngrok_domain = os.getenv('NGROK_DOMAIN', 'https://e7a9-213-192-2-118.ngrok-free.app')  # Fallback if not set
    
    print(f'Debug Info - Twilio Account SID: {account_sid[:5]}... (partial)')
    print(f'Debug Info - Twilio From Number: {from_whatsapp_number}')
    
    client = Client(account_sid, auth_token)
    
    try:
        if image_path and os.path.exists(image_path):
            # Construct a public URL using ngrok domain
            image_filename = os.path.basename(image_path)
            # Correct the path to match where profile images are saved
            public_image_url = f'{ngrok_domain}/static/images/profiles/{image_filename}'
            print(f'Attempting to send image with public URL: {public_image_url}')
            # Send image with caption
            message_response = client.messages.create(
                body=message,
                from_=from_whatsapp_number,
                to=f'whatsapp:{whatsapp_number}',
                media_url=[public_image_url]
            )
        else:
            print('No image provided or image file not found. Sending text message only.')
            # Send text message only
            message_response = client.messages.create(
                body=message,
                from_=from_whatsapp_number,
                to=f'whatsapp:{whatsapp_number}'
            )
        print(f'WhatsApp message sent successfully to {whatsapp_number}. SID: {message_response.sid}')
        return True
    except TwilioRestException as e:
        print(f'Error sending WhatsApp message: {str(e)}')
        return False

# Test the functionality
if __name__ == '__main__':
    # Test generating messages in different languages
    languages = ['English']  # Focus on English for now
    whatsapp_number = '+919987991854'  # Replace with actual number
    test_image_path = 'test/jason_statham.jpg'  # Specify the image path

    if os.path.exists(test_image_path):
        print(f'Processing image: {test_image_path}')
        profiling_data = predict_profiling_data(test_image_path)
        if 'error' not in profiling_data:
            print('Profiling data extracted successfully:', json.dumps(profiling_data, indent=2))
            for lang in languages:
                print(f'\nGenerating message in {lang}...')
                message = generate_profiling_message(profiling_data, lang)
                if not message.startswith('Error'):
                    send_whatsapp_message(message, whatsapp_number, test_image_path)
        else:
            print('Failed to extract profiling data:', profiling_data['error'])
    else:
        print(f'Test image not found at {test_image_path}') 