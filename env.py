import numpy as np
import ipdb

nb_tasks = 10
sz_personnalite = 3
nb_students = 3
mu = np.array([[-5, 3, 0],[4, 2, 1],[0, -3, -2]])
sigma = np.array([1,2,4])
nb_competences = 5

class Env:
    def __init__(self):
        1+1
    def reset(self, student_type):
        self.nb_comp=nb_competences
        self.student_type = student_type
        self.skills = np.random.rand(nb_competences) #todo: 0 ou 1, pas 0.42
        self.personnalite = np.zeros((sz_personnalite))
        self.current_task = np.zeros((nb_tasks))
        self.current_task[0] = 1
        self.task_id = 0
        
        self.req_skills = np.zeros((10,nb_competences))
        
        #TODO : initialiser la personnalite en fonction du type d'etudiant
        #par exemple, mixture de gaussiennes
        self.personnalite = mu[student_type] + np.random.normal(0, sigma[student_type], sz_personnalite)
        
        return self.get_state()
    
    def get_state(self):
        #ipdb.set_trace()
        return np.append(np.append(self.skills, self.personnalite),self.current_task)

    def play_action(self, action): #this function should return new state and reward. reward=1=>over
        if self.student_type == 0: #this student understands from any action, but needs the appropriate pre-requisits
            nb_lacunes = 0
            for i in range(self.nb_comp):
                if self.req_skills[self.task_id][i] > self.skills[i]:
                    nb_lacunes +=1
            if nb_lacunes == 1:
                return self.get_state(),1
            elif nb_lacunes == 1:
                return self.get_state(),1
            elif nb_lacunes <= 10 and action == 5: #hint
                return self.get_state(),1
            else:
                return self.get_state(),-0.1
        elif self.student_type == 1:
            return self.get_state(),-0.1
        else:
            return self.get_state(),-0.1
        return self.get_state(),-0.1

    def next_task(self):
        self.current_task[self.task_id] = 0 #on suppose les taches dans l'ordre
        if self.task_id == nb_tasks-1:
            return self.get_state(),0
        self.task_id += 1
        self.current_task[self.task_id] = 1
        return self.get_state(),1
