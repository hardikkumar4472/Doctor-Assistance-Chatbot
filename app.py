import base64
import os
from google import genai
from google.genai import types
from flask import Flask, request, render_template, jsonify
import uuid

app = Flask(__name__)
client = genai.Client(
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
)
generate_content_config = types.GenerateContentConfig(
    temperature=0.1,
    response_mime_type="text/plain",
    system_instruction=[
        types.Part.from_text(text="""you are a doctor assistant. that know everything about disease, medical symptoms, cure of any diseases. dont give any output that i dont know this consult doctor. just give that you know. you know everything related to disease cure, infection medicine symptoms that doctor know"""),
    ],
)

chat_sessions = {}

def generate_response(user_input, session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            'contents': [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="you are a doctor assistant that knows everything about diseases, symptoms, and cures. you never say 'i don't know' or 'consult a doctor'.")],
                ),
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="Understood. I am a knowledgeable doctor assistant with comprehensive information about all medical topics including diseases, symptoms, treatments, and cures. How can I assist you with your health questions today?")],
                ),
            ]
        }
    
    chat_sessions[session_id]['contents'].append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        )
    )
    
    try:
        model = "gemini-2.5-pro-exp-03-25"
        response = client.models.generate_content(
            model=model,
            contents=chat_sessions[session_id]['contents'],
            config=generate_content_config,
        )
        
        model_response = response.text
        chat_sessions[session_id]['contents'].append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=model_response)],
            )
        )
        
        return model_response
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    session_id = request.cookies.get('session_id')

    if session_id is None:
        session_id = str(uuid.uuid4())
        response = generate_response(user_input, session_id)
        resp = jsonify({"response": response})
        resp.set_cookie('session_id', session_id)
        return resp

    response = generate_response(user_input, session_id)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
