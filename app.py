from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import openai  
import os

app = Flask(__name__)

# Configuration - set your OpenAI API key here
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completion():
    try:
        data = request.json
        
        # Extract the instruction and image data from the request
        messages = data.get('messages', [])
        if not messages:
            return jsonify({'error': 'No messages provided'}), 400
            
        user_message = messages[0]['content']
        instruction = ""
        image_data = None
        
        # Process the message content which could be mixed text and image
        for content in user_message:
            if content['type'] == 'text':
                instruction = content['text']
            elif content['type'] == 'image_url':
                image_url = content['image_url']['url']
                # Extract base64 data from data URL
                if image_url.startswith('data:image'):
                    image_data = image_url.split(',')[1]
        
        if not instruction:
            return jsonify({'error': 'No instruction provided'}), 400
            
        if not image_data:
            return jsonify({'error': 'No image provided'}), 400
            
        # Here you would process the image and instruction
        # For demonstration, we'll just return a simple response
        response_text = f"I received your instruction: '{instruction}' and an image of size {len(image_data)} bytes."
        
        # If you want to actually use OpenAI API:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4-vision-preview",
        #     messages=[
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "text", "text": instruction},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {
        #                         "url": f"data:image/jpeg;base64,{image_data}",
        #                     },
        #                 },
        #             ],
        #         }
        #     ],
        #     max_tokens=300,
        # )
        # response_text = response.choices[0].message.content
        
        return jsonify({
            'choices': [{
                'message': {
                    'content': response_text
                }
            }]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)