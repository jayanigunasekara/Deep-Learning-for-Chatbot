#preprocess the data
#construct the seq to seq model
#create RNN encoder decoder
import numpy as np
import tensorflow as tf
import re

class ChatbotPreprocessing:
    def __init__(self):
        #import data set
        m_ChatLines = open('Data_set_lines.txt', encoding='utf-8',errors = 'ignore').read().split('\n')
        m_ChatConversations = open('Data_set_conversations1.txt', encoding='utf-8',errors = 'ignore').read().split('\n')
        
        #creating a line dictionary
        m_line_dictionary = {}
        for line in m_ChatLines:
            m_line = line.split('$')
            if len(m_line)== 3:
                m_line_dictionary[m_line[0]] = m_line[2]
        
        #creating a list of all the dialog IDs
        m_conversation_ID_dictionary = []
        for conversation in m_ChatConversations:
            m_conversation = conversation.split('$')[-1][1:-1].replace("'","").replace(" ","")   
            m_conversation_ID_dictionary.append(m_conversation.split(","))
        
        self.tuned_problems = []
        self.tuned_responses = []
        self.seperate_problems_and_responses(m_conversation_ID_dictionary, m_line_dictionary)
        self.frequency_word = {}
        
        self.problems_in_integers = {}
        self.responses_in_integers = {}
        
        self.frequency_of_words()
        self.map_strings_to_int()                       
        self.add_tokens()
        self.convert_int_ans_to_strings = {w_i : w for w, w_i in self.responses_in_integers.items()}            
                    
        #Adding the end of String token to the end of every response
        for i in range(len(self.tuned_responses)):
            self.tuned_responses[i]+='<EOS>'
          
        self.problems_converted_int = []
        self.responses_converted_int = []
        self.filter_with_OUT_token()
                    
        #sorting problems and responses by the length of the problem
        m_sorted_tuned_problems = []
        m_sorted_tuned_responses = []
        for length in range(1,25):
            for i in enumerate (self.problems_converted_int):
                if len(i[1]) == length:
                    m_sorted_tuned_problems.append(self.problems_converted_int[i[0]])
                    m_sorted_tuned_responses.append(self.responses_converted_int[i[0]])
                    
        
        #Setting the hyperparameters
        self.iterations =100
        self.group_size = 10
        self.learning_rate = 0.01
        self.lr_decay = 0.9
        self.minimum_lr= 0.0001
        self.probability_of_keeping = 0.5 
        m_num_layers = 3
        m_rnn_size = 800
        m_encoding_embedding_size = 512
        m_decoding_embedding_size = 512
           
        
        #session
        tf.reset_default_graph()
        self.session = tf.InteractiveSession()
                
        #loading input values to the model
        self.inputs, self.predictions, self.lr, self.selecting_probability = self.inputs_of_model()
                   
        #Set up the length of the sequence(problem)
        self.seq_length = tf.placeholder_with_default(25,None,name = 'sequence_length')     
        
        #input tensor 
        m_input_shape = tf.shape(self.inputs)      
        
        #Getting the training and test output
        m_training_output, self.test_output = self.seq2seq_model(tf.reverse(self.inputs,[-1]),
                                                                          self.predictions,
                                                                          self.selecting_probability,
                                                                          self.group_size,
                                                                          self.seq_length,
                                                                          len(self.responses_in_integers),
                                                                          len(self.problems_in_integers),
                                                                          m_encoding_embedding_size,
                                                                          m_decoding_embedding_size,
                                                                          m_rnn_size,
                                                                          m_num_layers,
                                                                          self.problems_in_integers)
        
        #calculate the error loss 
        #initiate Gradient Clipping
        with tf.name_scope("optimization"):
            self.error_loss = tf.contrib.seq2seq.sequence_loss(m_training_output,
                                                          self.predictions,
                                                          tf.ones([m_input_shape[0],self.seq_length]))
            m_optimize_value = tf.train.AdamOptimizer(self.learning_rate)
            m_gradients = m_optimize_value.compute_gradients(self.error_loss)
            m_gradients_clip = [(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable) for grad_tensor, grad_variable in m_gradients if grad_tensor is not None]
            self.gradient_clip_optimize = m_optimize_value.apply_gradients(m_gradients_clip)
            
        #split the data into training and validation
        m_split = int(len(m_sorted_tuned_problems)*0.15)
        self.problems_train = m_sorted_tuned_problems[ m_split:]
        self.responses_train = m_sorted_tuned_responses[ m_split:]
        self.problems_validation = m_sorted_tuned_problems[: m_split]
        self.responses_validation = m_sorted_tuned_responses[: m_split]
    
    #Getting seperately the problems and responses
    #remove the apostrophes        
    def seperate_problems_and_responses(self, conversationDict, lineDict): 
        m_problems = []
        m_responses = []
        for conversation in conversationDict:
            for i in range(len(conversation)-1):
                m_problems.append(lineDict[conversation[i]])
                m_responses.append(lineDict[conversation[i+1]])
                
        #adjust the problems removing the apostrophe 
        for problem in m_problems:
            self.tuned_problems.append(self.remove_abbreviations(problem))
            
        #adjust the responses removing the apostrophe 
        for response in m_responses:
            self.tuned_responses.append(self.remove_abbreviations(response))
            
    #create a dictionary of word and its frequency      
    def frequency_of_words(self):
        for problem in self.tuned_problems:
            for word in problem.split():
                if word not in self.frequency_word:
                    self.frequency_word[word]=1
                else:
                   self.frequency_word[word] = self.frequency_word[word]+1
        
        for response in self.tuned_responses:
            for word in response.split():
                if word not in self.frequency_word:
                    self.frequency_word[word]=1
                else:
                    self.frequency_word[word] = self.frequency_word[word]+1
    
    #creating the dictionary that map the words in problem to an integer iff it is greater than the frequency
    def map_strings_to_int(self):
        m_frequency = 5
        m_word_ID = 0
        for word,count in self.frequency_word.items():
            if count >= m_frequency:
                self.problems_in_integers[word]=m_word_ID
                m_word_ID = m_word_ID + 1
        m_word_ID = 0
        for word,count in self.frequency_word.items():
            if count >= m_frequency:
                self.responses_in_integers[word]=m_word_ID
                m_word_ID = m_word_ID + 1
     
    #adding the tokens to dictionary
    #<SOS> starting of the sentence
    #<EOS> ending of the sentence
    #<OUT> the unfrequently used words
    #<PAD>to map the problem word size to response word size
   
    def add_tokens(self):
    
        m_tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
        for token in m_tokens:
            self.problems_in_integers[token] = len(self.problems_in_integers)+1
                    
        for token in m_tokens:
            self.responses_in_integers[token] = len(self.responses_in_integers)+1 
      
    #Translate all the problems and responses into integers and repalcing 
    #all the words that were filtered by '<out>'
    def filter_with_OUT_token(self):
        for problem in self.tuned_problems:
            m_integer_values1 = []
            for word in problem.split():
                 if word not in self.problems_in_integers:
                     m_integer_values1.append(self.problems_in_integers['<OUT>'])
                 else:
                     m_integer_values1.append(self.problems_in_integers[word])
            self.problems_converted_int.append(m_integer_values1)      
                    
        for response in self.tuned_responses:
            m_integer_values2 = []
            for word in response.split():
                 if word not in self.responses_in_integers :
                     m_integer_values2.append(self.responses_in_integers ['<OUT>'])
                 else:
                     m_integer_values2.append(self.responses_in_integers [word])
            self.responses_converted_int.append(m_integer_values2)  
            
    #Remove the abbreviations and apostrophe
    def remove_abbreviations(self, text):
        text = text.lower()
        text = re.sub(r"i'm","i am",text)
        text = re.sub(r"he's","he is",text)
        text = re.sub(r"she's","she is",text)
        text = re.sub(r"it's","it is",text)
        text = re.sub(r"that's","that is",text)
        text = re.sub(r"this's","this is",text)
        text = re.sub(r"what's","what is",text)
        text = re.sub(r"\'ll","will",text)
        text = re.sub(r"\'ve","have",text)
        text = re.sub(r"\'re","are",text)
        text = re.sub(r"\'d","would",text)
        text = re.sub(r"won't","will not",text)
        text = re.sub(r"can't","cannot",text)
        text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]","",text)
        return text
    
    #create placeholders for inputs and predicted_output
    def inputs_of_model(self):
        m_inputs = tf.placeholder(tf.int32, [None,None],name = 'input')
        m_predicted_output = tf.placeholder(tf.int32,[None,None],name = 'target')
        m_lr = tf.placeholder(tf.float32,name = 'learning_rate')
        m_probability_holding = tf.placeholder(tf.float32,name = 'keep_prob')
        return m_inputs,m_predicted_output,m_lr,m_probability_holding 
    
    #predictions preprocess 
    def predictions_preprocess(self,predictions,int_value_word,grouping_size):       
        m_left_side = tf.fill([grouping_size,1],int_value_word['<SOS>'])
        m_right_side = tf.strided_slice(predictions,[0,0],[grouping_size, -1],[1,1])
        m_predictions_preprocess = tf.concat([m_left_side,m_right_side],1)
        return m_predictions_preprocess
                          
    #create the Encoder RNN layer
    def rnn_encoder(self,rnn_inputs,rnn_size,num_layers,prbability_holding,len_seq):
        m_lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        m_drop_lstm = tf.contrib.rnn.DropoutWrapper(m_lstm,input_keep_prob = prbability_holding)
        encoder_cell = tf.contrib.rnn.MultiRNNCell([m_drop_lstm]*num_layers)
        m_first,m_encoder_state =tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                         cell_bw = encoder_cell, 
                                                         sequence_length =len_seq ,
                                                         inputs = rnn_inputs,
                                                         dtype = tf.float32)
        return m_encoder_state
        
    #decode the training data 
    def training_data_decode(self,state_of_encoder,
                             decoder_cell,
                             decode_embd_input,
                             len_seq,
                             scope_decode,
                             output,
                             probability_holding,grouping_size):
        m_attention_state = tf.zeros([grouping_size,1,decoder_cell.output_size])
        m_id,m_values,m_function_score,m_function_const = tf.contrib.seq2seq.prepare_attention(m_attention_state,attention_option="bahdanau",num_units = decoder_cell.output_size) 
        m_decode_training = tf.contrib.seq2seq.attention_decoder_fn_train(state_of_encoder[0],
                                                                                  m_id,
                                                                                  m_values,
                                                                                  m_function_score,
                                                                                  m_function_const,      
                                                                                  name = "attn_dec_train")
        m_output,m_last_state,m_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                 m_decode_training,
                                                                                                                 decode_embd_input,
                                                                                                                 len_seq,
                                                                                                                 scope = scope_decode)
        m_drop_output_decode = tf.nn.dropout(m_output,probability_holding)
        return output(m_drop_output_decode)
    
    #Decode the test data 
    def test_data_decode(self,state_of_encoder,decoder_cell,decode_embd_matrix,sos_token_id, eos_token_id,len_max,num_words,len_seq,scope_decode,output,probability_holding,grouping_size):
        m_attention_state = tf.zeros([grouping_size,1,decoder_cell.output_size])
        m_id,m_values,m_function_score,m_function_const = tf.contrib.seq2seq.prepare_attention(m_attention_state ,attention_option="bahdanau",num_units = decoder_cell.output_size) 
        m_decode_test = tf.contrib.seq2seq.attention_decoder_fn_inference(output,
                                                                                  state_of_encoder[0],
                                                                                  m_id,
                                                                                  m_values,
                                                                                  m_function_score,
                                                                                  m_function_const,
                                                                                  decode_embd_matrix,
                                                                                  sos_token_id,
                                                                                  eos_token_id,
                                                                                  len_max,
                                                                                  num_words,
                                                                                  name = "attn_dec_inf")
        m_output_test,m_last_state,m_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                            m_decode_test,
                                                                                            scope = scope_decode)
        return m_output_test
    
    #RNN decoding the seq
    def rnn_decode(self,decode_embd_input,decode_embd_matrix,state_of_encoder,num_words,len_seq,rnn_size,num_layers,int_value_word,probability_holding, grouping_size):
        with tf.variable_scope("decoding") as decoding_scope:
            m_lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            m_drop_lstm = tf.contrib.rnn.DropoutWrapper(m_lstm,input_keep_prob = probability_holding)
            m_decoder_cell =  tf.contrib.rnn.MultiRNNCell([m_drop_lstm]*num_layers)
            m_weights = tf.truncated_normal_initializer(stddev=0.1)
            m_biases = tf.zeros_initializer()
            m_output = lambda x: tf.contrib.layers.fully_connected(x,
                                                                          num_words,
                                                                          None,
                                                                          scope = decoding_scope,
                                                                          weights_initializer = m_weights,
                                                                          biases_initializer = m_biases)
            m_output_training_data =self.training_data_decode(state_of_encoder,
                                                       m_decoder_cell,
                                                       decode_embd_input,
                                                       len_seq,
                                                       decoding_scope,
                                                       m_output,
                                                       probability_holding,
                                                       grouping_size)
            decoding_scope.reuse_variables() 
            m_output_test_data = self.test_data_decode(state_of_encoder,
                                               m_decoder_cell,
                                               decode_embd_matrix,
                                               int_value_word['<SOS>'],
                                               int_value_word['<EOS>'],
                                               len_seq -1,
                                               num_words,
                                               len_seq,
                                               decoding_scope,
                                               m_output,
                                               probability_holding,
                                               grouping_size)
            
        return m_output_training_data,m_output_test_data                                                                     
     
    #sequence to sequence model 
    def seq2seq_model(self,inputs,forecast,probability_holding,grouping_size,len_seq,responses_words,problems_words,embd_size_encode,embd_size_decode,rnn_size,num_layers,problems_in_int):
        m_embd_input_encode = tf.contrib.layers.embed_sequence(inputs,
                                                                  responses_words+1,
                                                                  embd_size_encode,
                                                                  initializer = tf.random_uniform_initializer(0,1))
        m_state_of_encoder = self.rnn_encoder(m_embd_input_encode,rnn_size,num_layers,probability_holding, len_seq)
        m_predictions_preprocess = self.predictions_preprocess(forecast,problems_in_int,grouping_size)

        m_embd_matrix_decode = tf.Variable(tf.random_uniform([problems_words+1,embd_size_decode],0,1))
        m_embd_input_decode = tf.nn.embedding_lookup(m_embd_matrix_decode,m_predictions_preprocess )
        output_training_data,output_test_data = self.rnn_decode(m_embd_input_decode,
                                                            m_embd_matrix_decode,
                                                            m_state_of_encoder,
                                                            problems_words,
                                                            len_seq,
                                                            rnn_size,
                                                            num_layers,
                                                            problems_in_int,
                                                            probability_holding,
                                                            grouping_size)
        return output_training_data,output_test_data
    
    #Padding the sequence with <PAD> token to get seq length to a fixed value
    def apply_padding(self, group_of_sequences,int_value_word):
        m_max_len_seq = max([len(seq) for seq in group_of_sequences])
        return [seq + [int_value_word['<PAD>']]* (m_max_len_seq-len(seq))for seq in group_of_sequences]           
                
    #Splitting the data into group of problems and responses
    def split_into_groups(self,problems, responses, grouping_size):
        for index in range(0,len(problems)// grouping_size):
            m_start_index = index * grouping_size
            m_problems_in_group = problems[m_start_index : m_start_index + grouping_size]
            m_responses_in_group = responses[m_start_index : m_start_index + grouping_size]
            m_problems_in_group_added_padding = np.array(self.apply_padding(m_problems_in_group,self.problems_in_integers))
            m_responses_in_group_added_padding =np.array(self.apply_padding(m_responses_in_group,self.responses_in_integers))
            yield m_problems_in_group_added_padding,m_responses_in_group_added_padding