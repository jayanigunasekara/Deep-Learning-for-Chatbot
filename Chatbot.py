# Building a chatbot with deep nlp
import numpy as np
import tensorflow as tf
from ChatbotPreprocessing import ChatbotPreprocessing

class Chatbot:
    def __init__(self):
        self.chatInitiate = 0
        
    def InitiateClass(self):
        self.myChatBot = ChatbotPreprocessing()
        #Loading the weights and running the session 
        m_check_point = "./chatbot_weights.ckpt"
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        m_saver = tf.train.Saver()
        m_saver.restore(self.session,m_check_point) 
        
        #converting the problems from Strings to encoding integers
    def convert_string_to_integers(self, problem,int_value_word):
        problem = self.myChatBot.remove_abbreviations(problem)
        return [int_value_word.get(word,int_value_word['<OUT>'])for word in problem.split()]
    
    def GetChatbotResponse(self, userInput):
        print('You:' + userInput)
        if self.chatInitiate == 0:
            self.chatInitiate = self.chatInitiate + 1
            return 'Hello! Welcome to MLDM Chatbot. How can I help you?'
        else:
            m_problem = userInput
            if m_problem == 'GoodBye':
                return 'GoodBye'
            m_problem = self.convert_string_to_integers(m_problem, self.myChatBot.problems_in_integers)
            m_problem = m_problem + [self.myChatBot.problems_in_integers['<PAD>']]*(20-len(m_problem))
            m_fake_group =  np.zeros((self.myChatBot.group_size,20))
            m_fake_group[0] = m_problem 
            m_response_output = self.session.run(self.myChatBot.test_output,{self.myChatBot.inputs:m_fake_group,self.myChatBot.selecting_probability:0.5})[0]
            m_response = ''
            for i in np.argmax(m_response_output,1):
                if self.myChatBot.convert_int_ans_to_strings == 'i':
                    token = 'I'
                elif self.myChatBot.convert_int_ans_to_strings[i] == '<EOS>':
                    token = '.'
                elif self.myChatBot.convert_int_ans_to_strings[i] == '<OUT>':
#                    return 'Please contact the administration.https://mldm.univ-st-etienne.fr/contacts.php'
                    token = 'Please contact the administration.https://mldm.univ-st-etienne.fr/contacts.php\n'
                else:
                    token = ' '+self.myChatBot.convert_int_ans_to_strings[i]
                m_response =m_response + token
                if token == '.':
                    break
            print('Chatbot:' + m_response)
           # print('Chatbot:' + m_response.replace('<PAD>', ''))
            return m_response.replace('<PAD>', '')
    
    
    


