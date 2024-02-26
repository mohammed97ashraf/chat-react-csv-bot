from flask import Flask, request, jsonify, render_template, url_for, redirect
import time
from langchain_helpper import create_embadding_from_csv,get_vector_db,react_agent_chat

app = Flask(__name__)


@app.route('/upload')
def upload_form():
    
    return render_template('upload_form.html')

@app.route('/chatbot/<botName>')
def chatbot(botName : str = None):
    return render_template('chat_box.html',botName=botName)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvFile' not in request.files:
        return "No file part"
    
    file = request.files['csvFile']
    
    print(request.form['botName'])
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.csv'):
        # Here you can add code to save or process the CSV file
        #file.save(file.filename)
        #return redirect(url_for('index', variable_name=f"{file.filename}"))
        create_embadding_from_csv(file_path=file.filename,bot_name=request.form['botName'])
        return redirect(url_for('chatbot',botName=request.form['botName']))
    else:
        return "Only CSV files are allowed"
    


@app.route("/chat", methods=['POST'])
def llm_dastion_chat():
    data = request.get_json()
    #bot_name = request.args.get('botName') 
    print(data.get("botName"))
    user_text = data.get('user_text')
    vec_db = get_vector_db(bot_name=data.get("botName"))
    ai_message = react_agent_chat(vectordb=vec_db,user_query=user_text,botname=data.get("botName"))
    return jsonify({"ai_message":ai_message,"results":"hello1"})

if __name__ == '__main__':
    app.run(debug=True)
