'''
This baseline implementation is developed in-house, based on the MILP-based algorithm for AMoD
systems as described in the following paper:
[1] M. Tsao, D. Milojevic, C. Ruch, M. Salazar, E. Frazzoli, and M. Pavone,
“Model Predictive Control of Ride-Sharing Autonomous Mobility-on-Demand Systems,” 
2019 International Conference on Robotics and Automation, pp. 6665–6671, IEEE, 2019.
'''

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

env = gp.Env(empty=True)
#env.setParam("OutputFlag",0)
env.start()

class optimization():
    def __init__(self, map, T, dt):
        '''
        map: a matrix containing the time it takes from one station to another
        for example:
        [[0,1,2],
        [1,0,3],
        [2,3,0]]
        means that the time it takes from station 0 to station 1 is 1, from 0 to 2 is 2, from 1 to 2 is three
        might not be symmetrical
        '''
        
        '''T : total planning horizon (number of time stamps)'''
        
        '''dt: length of each time stamp, like 1 sec'''
        
        self.map = map
        self.T = T
        self.dt = dt
        self.horizon = [i*dt for i in range(T)]
        self.N_station = len(map[0])
        self.N_T = len(self.horizon)
        
        self.model = gp.Model("model",env=env)
        self.r = self.model.addMVar((self.N_station,self.N_station,self.N_T),lb=0,vtype=GRB.INTEGER)
        self.x = self.model.addMVar((self.N_station,self.N_station,self.N_T),lb=0,vtype=GRB.INTEGER)
        self.d = self.model.addMVar((self.N_station,self.N_station,self.N_T),lb=0,vtype=GRB.INTEGER)

        ########################## graph connection constraints ####################################
        for i in range(self.N_station):

            for k in range(self.N_T):
                
                if k == 0:
                    continue
                
                lhs, rhs = 0, 0
                for j in range(self.N_station):
                    rhs += (self.r[i,j,k] + self.x[i,j,k])
                
                for j in range(self.N_station):
                    if j == i:
                        prev_k = int(k - 1)
                    else:
                        prev_k = int(np.round(k-self.map[j,i]/self.dt))

                    if prev_k >= 0:
                        lhs += (self.r[j,i,prev_k]+self.x[j,i,prev_k])
            
                self.model.addConstr(rhs == lhs)
        
        ########################## measure deliveries ####################################
        for i in range(self.N_station):
            for j in range(self.N_station):
                for k in range(self.N_T):
                    lhs = 0
                    for tau in range(k+1):
                        lhs += self.x[i,j,tau]
                    
                    self.model.addConstr(self.d[i,j,k]==lhs)

        ################################## objectives ####################################
        f1, f2, f3 = 0, 0, 0
        c1, c2, c3 = 1, 1, 1
        for i in range(self.N_station):
            for j in range(self.N_station):
                for k in range(self.N_T):
                            
                    f1 += (-self.d[i,j,k])
                    f2 += self.map[i,j]*self.r[i,j,k]
                    f3 += self.map[i,j]*self.x[i,j,k]

        f = c1*f1 + c2*f2 + c3*f3
        self.model.setObjective(f,GRB.MINIMIZE)
        
        self.init_step_constraints = [None] * self.N_station
        self.d_constraints = [None] * (self.N_T*self.N_station*self.N_station)

    def update(self,request,vehicles):
        '''
        request: a 2-D array that features demandings of passengers
        for example:
        [[0,2],[0,3],[4,5]]
        means that there are three passengers, the first passengers from station 0 to station 2,
        the second from station 0 to station 3,
        the third from station 4 to station 5
        '''

        '''
        vehicles: 1-D array that implies current locations of all vehicles
        for example: [5,1]
        means that the first vehicle is at station 5 and the second is at station 1
        '''
        self.request = request
        self.vehicles = vehicles
        self.N_vehicles = len(vehicles)
        self.vehicle_dict = np.zeros(self.N_station)
        for j in range(self.N_vehicles):
            self.vehicle_dict[self.vehicles[j]] += 1
        
        
        ########################## initial constraints ####################################
        for i in range(self.N_station):

            lhs, rhs = 0, 0
            for j in range(self.N_station):
                rhs += (self.r[i,j,0] + self.x[i,j,0])
            
            lhs = self.vehicle_dict[i]
            
            if self.init_step_constraints[i] is not None:
                self.model.remove(self.init_step_constraints[i])
                
            self.init_step_constraints[i] = self.model.addConstr(rhs == lhs)
            
        ########################## deliveries smaller than demandings ######################
        for i in range(self.N_station):
            for j in range(self.N_station):
                for k in range(self.N_T):
                    req_count = 0
                    for req in self.request:
                        if req[0] == i and req[1] == j:
                            req_count += 1
                    
                    index = i*(self.N_station*self.N_T) + j*(self.N_T) + k
                    if self.d_constraints[index] is not None:
                        self.model.remove(self.d_constraints[index])
                    
                    self.d_constraints[index] = \
                        self.model.addConstr(self.d[i,j,k]<=req_count)
    
    def solve(self,):
        '''call this function to return the match'''

        self.model.optimize()
        r_X, x_X = self.r.X, self.x.X
        route_list = []
        for i in range(self.N_station):
            for j in range(self.N_station):
                while r_X[i,j,0] > 0:
                    route = self.get_route(r_X,x_X,i,j)
                    route_list.append(route)

        match = []
        r_claimed = [False]*len(route_list)
        p_accepted = [False]*len(self.request)
        for i,start in zip(range(len(self.vehicles)),self.vehicles):
            p_list = []
            for j,route in zip(range(len(route_list)),route_list):
                
                if route[0][0] == start and not r_claimed[j]:
                    r_claimed[j] = True
                    
                    for link in route:
                        if link[3] == 'r':
                            continue
                        
                        depart, des = link[0], link[1]
                        for k,p in zip(range(len(self.request)),self.request):
                            if depart == p[0] and des == p[1] and not p_accepted[k]:
                                p_accepted[k] = True
                                p_list.append(k)
                                break
                    
                    break
                    
            match.append(p_list)
        
        '''
        match: a 2-D list that define the match between vehicles and passengers
        for example:
        [[0,1],[2]]
        means that the first vehicle is assigned with passengers 0 and 1, 
        while the second vehicle is assigned with passenger 2.
        note that one vehicle might be assigned with multiple passengers.
        '''
        
        '''
        route_list:
        a list of routes by all vehicles for inspection
        each route is a 2-D list that is performed by one vehicle
        each link of the route is a Quaternion [a,b,c,d],
        which means that the vehicle depart from station 'a' to station 'b'
        at c_th time stamp. 
        if d == 'r', the vehicle is empty.
        if d == 'x', the vehicle is occupied.
        '''
        
        return match, route_list
    
    def get_route(self,r_X,x_X,i,j):
        '''
        inner function to return the route
        no need to use this method from outside
        '''
        k = 0
        route = []
        r_X[i,j,0] -= 1
        route.append([i,j,k,"r"])
        while True:
            
            if j == i:
                k += 1
            else:
                k += int(self.map[i,j]/self.dt)
            i = j
            if k >= self.N_T:
                break

            for j in range(self.N_station):

                if r_X[i,j,k] >0:
                    r_X[i,j,k] -= 1
                    route.append([i,j,k,"r"])
                    break

                elif x_X[i,j,k] >0:
                    x_X[i,j,k] -= 1
                    route.append([i,j,k,"x"])
                    break
        
        return route

if __name__ == "__main__":
    "example of how to use this module"
    "would be better if the map size and the horizon are smaller than 20"
    map_height = 5
    map_width = 5
    map_size = map_height*map_width
    map = np.zeros((map_size,map_size))
    for i in range(map_size):
        for j in range(map_size):
            h_i = i//map_width
            w_i = i%map_width
            h_j = j//map_width
            w_j = j%map_width
            dis = np.abs(h_i-h_j) + np.abs(w_i-w_j)
            map[i,j] = dis
            
    t1 = time.time()
    # define map, the initialization step might take dozen of seconds depend on size of the problem
    optimizer = optimization(map,10,0.1) 
    t2 = time.time()
    optimizer.update([[4,12],[0,4]],[1,12]) # update request and positions of vehicles
    t3 = time.time()
    match,route = optimizer.solve() # obtain results
    t4 = time.time()
    
    print(match)
    print(route)
    
    optimizer.update([[4,12],[7,8]],[12,12]) # update request and positions of vehicles
    t5 = time.time()
    match,route = optimizer.solve() # obtain results
    t6 = time.time()
    
    print(match)
    print(route)

    print("Time taken for initialization: ",t2-t1)
    print("Time taken for update: ",t3-t2)
    print("Average time taken for solving: ",(t4-t3+t6-t5)/2)
