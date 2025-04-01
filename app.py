import os
from flask import Flask, request, render_template, jsonify
import google.generativeai as genai
import uuid
app = Flask(__name__)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  
    generation_config=generation_config,
    system_instruction="""You are a doctor assistant that knows everything about diseases, medical symptoms, 
    and cures of any diseases. Don't say "I don't know" or "consult a doctor". Provide the medical 
    information you have. You know everything related to disease cure, infection, medicine, and symptoms 
    that doctors know."""
)

chat_sessions = {}

def generate_response(user_input, session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = model.start_chat(history=[])
    
    try:
        response = chat_sessions[session_id].send_message(user_input)
        return response.text
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
