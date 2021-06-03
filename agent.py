import numpy as np
from numpy import genfromtxt
import pandas as pd
import threading
import time

class QAgent():
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, alpha, gamma, location_to_state, actions, rewards, state_to_location, Q, terminal_set_status):
        
        self.gamma = gamma  
        self.alpha = alpha 
        self.location_to_state = location_to_state
        self.actions = actions
        self.rewards = rewards
        self.state_to_location = state_to_location
        self.Q = Q
        self.terminal_set_status = terminal_set_status
        

    # Training the agent in the environment
    def training(self, end_location, iterations):
        rewards_new = np.copy(self.rewards)

        for i in end_location :
            ending_state = self.location_to_state[i]
            rewards_new[ending_state, ending_state] = 3000
        
        for i in range(iterations):
            current_state = np.random.randint(0,35) 
            playable_actions = []

            for j in range(35):
                if rewards_new[current_state,j] > 0:
                    playable_actions.append(j)
        
            if(len(playable_actions)>0) :
                next_state = np.random.choice(playable_actions)
                TD = rewards_new[current_state,next_state] + self.gamma * self.Q[next_state, np.argmax(self.Q[next_state,])] - self.Q[current_state,next_state]
                
                self.Q[current_state,next_state] += self.alpha * TD
        # Return the new Q values computed
        return self.Q

    # Getting the optimal route
    def get_optimal_route(self, start_location, end_location, next_location, route, Q, charge_time, red_factor):
        my_wait = 0
        print("Optimal Route")
        i=0
        while(next_location not in end_location):
            starting_state = self.location_to_state[start_location]
            next_state = np.random.choice(np.where(Q[starting_state] == Q[starting_state].max())[0])
            next_location = self.state_to_location[next_state] 
            if(next_location not in route):    
                route.append(next_location)     # Storing the route
            start_location = next_location
            i += 1
            if(i>1000):
                break
        
        flag = False
        original_reward = 0
        for i in range(len(route)-1):
            starting_state1 = self.location_to_state[route[i]]
            ending_state1 = self.location_to_state[route[i+1]]
            self.rewards[starting_state1, ending_state1] -= red_factor      # Reducing Reward for that route as Congestion has increased
            if(self.rewards[starting_state1, ending_state1] < 0) : 
                flag = True
                original_reward = self.rewards[starting_state1, ending_state1] + red_factor
                self.rewards[starting_state1, ending_state1] = 0
        if(route[-1] not in end_location):
            print("Failed!")
        else :
            print(route)
            print("Charging Time = ", charge_time)
            if(self.terminal_set_status[route[-1]] == 0) :      # Computing On Charging Time at Station
                print("Charging Cost = Rs ", charge_time * 10)      
            else :
                print("Charging Cost = Rs ", charge_time * 10 + (10*self.terminal_set_status[route[-1]]))
            my_wait = self.terminal_set_status[route[-1]]
            self.terminal_set_status[route[-1]] = self.terminal_set_status[route[-1]] + charge_time     # Changing the Congestion Status of the Allotted Station
            
            # Reducing the Congestion Status of the occupied station while each unit of battery gets charged
            if(my_wait) :
                time.sleep(my_wait)
            while(charge_time>0) : 
                time.sleep(1)
                self.terminal_set_status[route[-1]] = self.terminal_set_status[route[-1]] - 1
                charge_time = charge_time - 1 
        
        # Reverting back to Original Reward after Charging is over
        for i in range(len(route)-1):
            starting_state1 = self.location_to_state[route[i]]
            ending_state1 = self.location_to_state[route[i+1]]
            if(not flag):
                self.rewards[starting_state1, ending_state1] += red_factor
            else :
                self.rewards[starting_state1, ending_state1] = original_reward
        return


# For Location Name to State mapping
location_to_state = {
    'Location1' : 0,
    'Location2' : 1,
    'Location3' : 2,
    'Location4' : 3,
    'Location5' : 4,
    'Location6' : 5,
    'Location7' : 6,
    'Location8' : 7,
    'Location9' : 8,
    'Location10' : 9,
    'Location11' : 10,
    'Location12' : 11,
    'Location13' : 12,
    'Location14' : 13,
    'Location15' : 14,
    'Location16' : 15,
    'Location17' : 16,
    'Location18' : 17,
    'Location19' : 18,
    'Location20' : 19,
    'Location21' : 20,
    'Location22' : 21,
    'Location23' : 22,
    'Location24' : 23,
    'Location25' : 24,
    'Location26' : 25,
    'Location27' : 26,
    'Location28' : 27,
    'Location29' : 28,
    'Location30' : 29,
    'Location31' : 30,
    'Location32' : 31,
    'Location33' : 32,
    'Location34' : 33,
    'Location35' : 34
}

# Define the rewards
rewards = genfromtxt('/Users/moyeensarfaraj/Desktop/Major_Project/Reward_Matrix_csv.csv', delimiter=',')
rewards = rewards*1200
#Define actions
actions = {'left', 'right', 'up', 'down'}

# For mapping State to Location Names
state_to_location = dict((state,location) for location,state in location_to_state.items())

# Initialise parameters
gamma = 0.75 # Discount factor 
alpha = 0.9 # Learning rate 
red_factor = 400 # Reducing factor
# Initialising Q-Values
Q = np.array(np.zeros([35,35]))

# Setting the Location of the Charging Stations
terminal_set = []
for i in range(3):
    terminal_set.append('Location'+input("Enter the Location Number of the Charging Station : Location "))
#Initialising the Congestion Status of each Charging Status
terminal_set_status = {}
for i in terminal_set:
    terminal_set_status[i] = 0

# Creating an agent object
qagent = QAgent(alpha, gamma, location_to_state, actions, rewards,  state_to_location, Q, terminal_set_status)

# Driving loop
i = 0
while(1):
    print("\n-----------------------------------------------------------------------------------")
    print("1. View Congestion Status.")
    print("2. Request Charging.")
    print("3. Exit.")
    choice = int(input("Enter your choice : "))
    print("\n")
    if(choice == 3):
        break
    # For viewing the Live Congestion Status of each Charging Station
    elif(choice == 1) :
        print("Congestion Status of Charging Stations : ")
        print('Station \t Waiting')
        for q in qagent.terminal_set_status:
            print(q,"\t", qagent.terminal_set_status[q])
        continue
    # For Requesting Charging
    else :
        if(i>=10) :
            print("Limit of 10 exceeded!!!")
        else : 
            # Taking the Vehicle related information from Vehicle Driver
            start_location = 'Location'+input("Enter Location Number of this EV : ")
            if start_location in ['Location9', 'Location11', 'Location13', 'Location16', 'Location18', 'Location23', 'Location25', 'Location27'] :
                print("Invalid Location")
                continue
            vehicle_class = int(input("Enter the Class of Vehicle (1. Two wheeler 2. Light Motor Vehicle 3. Heavy Vehicle) : "))
            current_charge = int(input("Enter the current units of Charge left : "))

            # Computing charging time based on Vehicle Class
            if(vehicle_class == 1) :
                battery_capacity = 25
            elif(vehicle_class == 2) :
                battery_capacity = 60
            else :
                battery_capacity = 100
            charging_time = battery_capacity - current_charge

            # Training based on the instantaneous congestion
            route = [start_location]
            next_location = start_location
            print("For Vehicle ", (i+1))
            Qt = qagent.training(terminal_set ,15000)   #Training

            # Thread created for the Vehicle which finds the optimal route
            t1 = threading.Thread(target=qagent.get_optimal_route, args=(start_location, terminal_set, next_location, route, Qt, charging_time, red_factor+charging_time))
            t1.start()
            time.sleep(2)

            print("-----------------------------------------------------------------------------------\n")
            i = i+1
