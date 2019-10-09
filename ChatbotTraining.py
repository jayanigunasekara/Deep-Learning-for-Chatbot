import tensorflow as tf
import time
from ChatbotPreprocessing import ChatbotPreprocessing

class ChatbotTraining:
    def __init__(self):
        self.process_data = ChatbotPreprocessing()
        m_Group_Index_When_Check_Training_Lost = 10
        m_Group_Index_When_Check_Validation_Lost = ((len(self.process_data.problems_train )//self.process_data.group_size//2)-1)
        m_sum_error_loss_training = 0
        m_error_loss_validation = []
        m_check_early_stop = 0
        m_stop_early = 10
        m_weights_checkpoint = "./chatbot_weights.ckpt"
        self.process_data.session.run(tf.global_variables_initializer())
        for i in range (1,self.process_data.iterations+1):
            for group_index, (problems_in_group_added_padding,responses_in_group_added_padding) in enumerate(self.process_data.split_into_groups(self.process_data.problems_train ,self.process_data.responses_train,self.process_data.group_size)):
                m_starting_time = time.time()
                first_element,m_error_loss_of_training_group = self.process_data.session.run([self.process_data.gradient_clip_optimize,self.process_data.error_loss],{self.process_data.inputs:problems_in_group_added_padding,
                                                           self.process_data.predictions:responses_in_group_added_padding,
                                                           self.process_data.lr:self.process_data.learning_rate,
                                                           self.process_data.seq_length:responses_in_group_added_padding.shape[1],
                                                           self.process_data.selecting_probability: self.process_data.probability_of_keeping})
            
    
                m_sum_error_loss_training =m_sum_error_loss_training + m_error_loss_of_training_group
                m_ending_time = time.time()
                m_grouping_time = m_ending_time - m_starting_time
                if group_index % m_Group_Index_When_Check_Training_Lost ==0:
                    print('Iteration: {:>3}/{},Group:{:>4}/{},Training loss error: {:>6.3f},Training time on 100 groups:{:d} seconds'.format(i,
                                                                                                                                          self.process_data.iterations,
                                                                                                                                          group_index,
                                                                                                                                          len(self.process_data.problems_train )//self.process_data.group_size,
                                                                                                                                          m_sum_error_loss_training/m_Group_Index_When_Check_Training_Lost,
                                                                                                                                          int(m_grouping_time*m_Group_Index_When_Check_Training_Lost)))
                    m_sum_error_loss_training =  0
                if group_index % m_Group_Index_When_Check_Validation_Lost ==0 and group_index > 0:
                    m_sum_error_loss_validation = 0
                    m_starting_time = time.time()
                    for group_index_validation, (problems_in_group_added_padding,responses_in_group_added_padding) in enumerate(self.process_data.split_into_groups(self.process_data.problems_validation,self.process_data.responses_validation,self.process_data.group_size)):
                        m_error_loss_of_validation_group = self.process_data.session.run(self.process_data.error_loss,{self.process_data.inputs:problems_in_group_added_padding,
                                                                              self.process_data.predictions:responses_in_group_added_padding,
                                                                              self.process_data.lr:self.process_data.learning_rate,
                                                                              self.process_data.seq_length:responses_in_group_added_padding.shape[1],
                                                                              self.process_data.selecting_probability: 1})
            
                        m_sum_error_loss_validation = m_sum_error_loss_validation + m_error_loss_of_validation_group 
                    m_ending_time = time.time()
                    m_grouping_time = m_ending_time - m_starting_time
                    avg_error_loss_validation = m_sum_error_loss_validation/(len(self.process_data.problems_validation) / self.process_data.group_size)
                    print('error loss-validation:{:>6.3f},Group validation time :{:d} seconds'.format(avg_error_loss_validation,int(m_grouping_time )))
                    self.process_data.learning_rate *= self.process_data.lr_decay
                    if self.process_data.learning_rate < self.process_data.minimum_lr:
                        self.process_data.learning_rate = self.process_data.minimum_lr
                    m_error_loss_validation.append(avg_error_loss_validation)
                    if avg_error_loss_validation <= min(m_error_loss_validation):
                        print('better trained')
                        m_check_early_stop = 0
                        saver = tf.train.Saver()
                        saver.save(self.process_data.session,m_weights_checkpoint)
                    else:
                        print("Cannot be trained ,need time to practice")
                        m_check_early_stop += 1
                        if m_check_early_stop ==m_stop_early :
                            break
            if m_check_early_stop == m_stop_early :
                print("cannot trained anymore")
                break