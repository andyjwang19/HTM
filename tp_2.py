import numpy as np
import matplotlib.pyplot as plt

import math
import argparse

import random
import copy

import sys


#HYPERPARAMETERS

num_of_vectors = 9
vector_output_grid = math.ceil(num_of_vectors **0.5)
vector_len = 100

number_of_connections = 15
weight_change = 0.005
net_width = 70
net_height = net_width
radius_of_inhibition = 6
num_of_epochs = 100
total_active = 50

number_of_distal = 4
total_cells = number_of_distal


#IF THINGS
change_factor = 2
run_training = True


class column(object):
    def __init__(self, x_value, y_value, index_vector):
        self.x = x_value
        self.y = y_value

        self.value = 0
        self.connections = []
        self.weights = []

        self.distal = []
        self.distal_weights=[]

        #Each cell contains the coordinates of the column that sent the distal signal
        self.cells = [[] for x in range(0, total_cells)]




        for c in range(0,number_of_distal):
            self.distal.append([random.randint(0,net_width-1), random.randint(0,net_height-1)])
            #Not normal distribution... genconnnections is...
            self.distal_weights.append(0.5)
        #print(self.distal)
        #print(self.distal_weights)


    #def generateConnections(self, index_vector):
        #has to be even
        self.connections = np.random.choice(index_vector, number_of_connections, False)
        for i in range(0, number_of_connections):
            self.weights.append(np.random.uniform(0,1))
            #self.weights.append(np.random.uniform(0.1,1))
    #   return self.connections, self.weights

    def calculateValue(self, input_vector):
        self.value = 0
        for i in range(0,len(self.connections)):
            if self.weights[i] > 0.1:
                self.value += input_vector[self.connections[i]] * self.weights[i]
       #print(str(self.x),str(self.y))
       #print(self.value)
        return self.value

#    def genDistal(self):

    def activateCells(self,x,y):
        for b in range(0, len(self.cells)):
            if self.cells[b] == []:
                self.cells[b] =[x,y]


    def distalSignal(self):
       for f in range(0,len(self.distal)):
            if self.distal_weight[f] > 0.1:
                net[self.distal[f]].activateCells(self.x,self.y)


    def updateDistal(self, x, y):
        for f in range(0, len(self.distal)):
            if self.distal[f] == [x,y]:
                self.distal_weights[f] += weight_change

    def fire(self, net, input_vector, temp, epoch):

        #update weights for lateral connects
        #cells contain the coordinates of sending column if predicted
        for cell in self.cells:
            if cell == []:
                break
            print(self.x,self.y)
            net[cell[0]][cell[1]].updateDistal(self.x,self.y)

        #print(str(self.x), str(self.y))

        small_input = [input_vector[self.connections[x]] for x in range(0, len(self.connections))]
        for i in range(0,len(small_input)):
            if small_input[i] == 0 and self.weights[i]- weight_change > 0:
                self.weights[i] -= weight_change
            if small_input[i] == 1 and self.weights[i] + weight_change < 1:
                self.weights[i] += weight_change
       #if temp == 3:
       #    print("Column: "+ str(self.x) +',' + str(self.y) + " epoch: " + str(epoch) +  " weights: ")
       #    print(self.weights)
       #    print(small_input)

        #send distal signals
        for f in range(0,len(self.distal)):
            if self.distal_weights[f] > 0.1:
                print(str(len(net)), str(len(net[0])))
                print(self.distal[f][0])
                print(self.distal[f][1])
                net[self.distal[f][0]][self.distal[f][1]].activateCells(self.x,self.y)




def sdr(value_net):
    #problem if valuenet is purely 0s
    flat = sum(value_net, [])
    count = 0
    while max(flat) != 0 and count < total_active:
        flat = sum(value_net, [])
        max_num = np.argmax(flat)
        y = int(max_num%net_width)
        x = int(math.floor(max_num/net_width))


        for q in range(x-radius_of_inhibition, x + 1+ radius_of_inhibition):
            for t in range(y-radius_of_inhibition, y + 1+ radius_of_inhibition):
                if q >= 0 and t >= 0 and q < net_height and t < net_width:
                    value_net[q][t] = 0
        value_net[x][y] = -1
        count += 1
    return value_net


def step(net, input_vector, index_vector, epoch, test, plot):
    if not plot:
        print(str(epoch+1) + '/' + str(num_of_epochs))

    value_net = []
    iteration = 0
    for ar in net:
        value_net.append([])
        for obj in ar:
        #   if epoch == 0:
        #       obj.generateConnections(index_vector)
        #       obj.genDistal()
            value_net[iteration].append(obj.calculateValue(input_vector))
        iteration+= 1
    #print(value_net)

   #if epoch == num_of_epochs - 2 :
   #    global old_vn
   #    print("d")
   #    old_vn = copy.deepcopy(value_net)

   ##prints difference between value nets
   #if test:
   #    print(epoch)
   #    new_vn = copy.deepcopy(value_net)
   #    for x in range(0, len(new_vn)):
   #        for y in range(0, len(new_vn[x])):
   #            new_vn[x][y] = new_vn[x][y] - old_vn[x][y]
   #    print(new_vn)


    #creates sparse distributed representation
    sdr_net = sdr(value_net)
    #print(sdr_net)


    #trains if test is off
    if not test:
        temp = 0
        for x in range(0, len(sdr_net)):
           for y in range(0, len(sdr_net[x])):
               if sdr_net[x][y] == -1:
                   net[x][y].fire(net, input_vector, temp, epoch)
                   temp += 1


    #plots if this is the last training iteration or if testing
    if plot:
        plt.axis([-1,net_width+1,-1,net_height])
        ax = plt.subplot(vector_output_grid,vector_output_grid,plot)
        ax.set_title("Vector: " + str(plot))
        for x in range(0, len(sdr_net)):
            for y in range(0, len(sdr_net[x])):
                if sdr_net[x][y] == -1:
                    if test:
                        ax.plot([x],[y],'b^')
                    else:
                        ax.plot([x],[y], 'ro')

    return net


def reverseArray(ar):
    result = []
    for num in ar:
        if num == 1:
            result.append(0)
        else:
            result.append(1)
    return result

def generateVector(length):
    result = []
    for i in range(0, length):
        result.append(int(random.getrandbits(1)))
    return result


def main():
    parser = argparse.ArgumentParser(description = "runs spatial pooling portion of htm model")
    parser.add_argument("--a", default = False)
    args = parser.parse_args()


    input_vector = []
    test_vector = []
    for w in range(0,num_of_vectors):
        input_vector.append(generateVector(vector_len))
        test_vector.append(copy.copy(input_vector[w]))
        for i in range(0, change_factor):
            test_vector[w][i] = 1 - test_vector[w][i]



    index_vector = [i for i in range(0,vector_len)]

    net = [[column(i, q, index_vector) for i in range(0,net_width)] for q in range(0,net_height)]
    #print(net)


    if run_training:
        for epoch in range(0, num_of_epochs):
            for i in range(0, num_of_vectors):
                if epoch % num_of_vectors == i:
                    net = step(net, input_vector[i], index_vector, epoch, False, 0)
    else:
        epoch = 0


    for i in range(0, num_of_vectors):

        step(net,input_vector[i],index_vector, epoch, False, i+1)
        if epoch == 0:
            epoch+= 1
        step(net, test_vector[i], index_vector, epoch, True, i+1)


    plt.show()


if __name__ == "__main__":
    main()
