from flask import Flask, render_template, request
from ChatbotTraining import ChatbotTraining
from Chatbot import Chatbot

app = Flask(__name__)

IsTrainingComplete = 0;
myChat = Chatbot()

@app.route('/')
@app.route('/Home')
def index():
	return render_template('Home.html')

@app.route('/training', methods=['POST'])
def chattraining():
    print("training pressed")
    ChatbotTraining()
    return render_template('Home.html', trainingerror='Training complete!')

@app.route('/startchatbot', methods=['POST'])
def startchatbot():
    print("start chatbot pressed")
    myChat.InitiateClass()
    return render_template('Chatbot.html')
	  #else:
	 	 #return render_template('Home.html', chatboterror='Please train the chatbot first!')

@app.route('/get')
def GetBotResponse():

    m_user_Input = request.args.get('msg')
    m_bot_Response = myChat.GetChatbotResponse(m_user_Input)
    return m_bot_Response

if __name__=='__main__':
	#app.run(debug=True,port=5002)
	app.run()
