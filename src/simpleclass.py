import Classes as cl
import copy
import math
import itertools as it
import collections
import random

class Robot(object):
    idx = it.count(0)
    def __init__(self,position_x,position_y,name):
        self.position_x = position_x
        self.position_y = position_y
        self.name = name
        self.location = (position_x,position_y)
        self.speed = 0
        self.emotion = "anxious"
        self.health = 100
        self.idx = self.idx.next()
        #0 = N 1 = E 2 = S 3 = W
        self.orientation = random.sample(("W","N","S","E"),1)

class Board(object):
    def __init__(self,size_x,size_y):
        self.size_x = size_x
        self.size_y = size_y
        self.area = size_x * size_y
        #self.rounds = rounds
        #self.damage = damage
    #def build_board(self):
    #    area = self.size**2
        #randomly populate robots

    def build_robots(self,num_robots):
        #TODO: add only unique locations
        #populate board with robots
        #locations = all.unique((random.randint(0,self.size_x),(random.randint(0,self.size_y))
        robots = [Robot(random.randint(0,self.size_x),random.randint(0,self.size_y),i) for i in xrange(0,num_robots)]
        return robots


class Round(object):
    def __init__(self,damage,total_damage,num_robots,roundinfo):
        self.damage = damage
        self.total_damage = total_damage
        self.num_robots = num_robots
        self.roundinfo = roundinfo

newGame = Board(9,9)
robots = newGame.build_robots(4)
print [robots[i].location for i in xrange(0,len(robots))]
print [robots[i].orientation for i in xrange(0,len(robots))]
#trundlebot = Robot.build_robot
#print trundlebot.health
