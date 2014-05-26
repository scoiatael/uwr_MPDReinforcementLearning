#!/bin/python
Ne = 15
INFINITY = 10000

import argparse 
import readline
import cmd
import os.path
from random import random
from functools import reduce
from operator import *
from sys import exit

def runAgent(agent):
    c = cmd.Cmd()
    c.postcmd = (lambda x, y: print(agent.printable()) or False)
    c.do_n = (lambda x: agent.next)
    c.do_q = (lambda x: exit(0))
    c.cmdloop()

class Tile:
    '''Single tile of world'''
    def __init__(self, t, v):
        state, value = t.split(':')
        self.state=state
        if(value != ""):
            self.value = int(value)
        if(not type(v) is float):
            raise ValueError("Value for tile not provided");

    def value(self):
        return self.value

    def isTerminal(self):
        return self.state=="G"
    def isStarting(self):
        return self.state=="S"
    def isForbidden(self):
        return self.state=="F"
    def isSpecial(self):
        return self.state=="B"

class Board:
    def __init__(self, val, string=[]):
        self.board = [[Tile(t, val) for t in lne.split()] for lne in string]

    def at(self, p):
        return self.board[p.imag][p.real]

    def findStarting(self):
        for ln in range(len(self.board)):
            for t in range(len(self.board[ln])):
                if self.board[ln][t].state == "S":
                    return (t,ln)

class Move:
    PointConversion = { "U":0+1j, "D":0-1j, "L":-1-0j, "R":1-0j }

    def __init__(self, state):
        if(state in Move.PointConversion.keys()):
            self.state=state
        else:
            raise ValueError("Bad move definition")
            
    def isUp(self):
        return self.state=="U"
    def isDown(self):
        return self.state=="D"
    def isRight(self):
        return self.state=="R"
    def isLeft(self):
        return self.state=="L"

    def toPoint(self):
        return Move.PointConversion[self.state]

    def possibleOutcomes(self):
        if(self.isUp() or self.isDown()):
            return [self] + [Move(x) for x in "R L".split()]
        return [self] + [Move(x) for x in "U D".split()]

allMoves = [Move(x) for x in "U D R L".split()]

class World:
    '''Description of world agent acts in'''
    def __init__(self, N,M,a,b,r,string=[]):
        self.N=N
        self.M=M
        self.a=a
        self.b=b
        self.board=Board(r, string)
        if(abs(a+b+b - 1) > 0.01):
            raise ValueError("Bad probabilities")

    def simulateMove(self, state, move, sim = True):
        pAndR = [ (ns, self.board.at(ns)) for ns in [ state + p.toPoint() for p in move.possibleOutcomes()] ]
        pAndR = [ (state, self.board.at(state).value) if b.isForbidden() else (s,b) for s,b in pAndR ]
        pAndR = [ ( pAndR[0][0],pAndR[0][1], self.a ), ( pAndR[1][0], pAndR[1][1], self.b ), ( pAndR[2][0], pAndR[2][1], self.b ) ]
        if( not sim ):
            return pAndR
        r = random() - self.a
        if(r < 0):
            return pAndR
        if(r < self.b): 
            return [pAndR[1], pAndR[0], pAndR[2]]
        return [pAndR[2], pAndR[0], pAndR[1]]


class ValueIterateAgent:
    '''Agent using value iteration algorithm'''
    def __init__(self, d, world):
        self.world = world
        self.d = d
        self.U = [[ 0 for x in range(world.N)] for y in range(world.M)]

    def next(self):
        self.U = [[self.world.board.at(complex(x,y)).value 
                + self.d * max(reduce(add, [ v*p in s,v,p in self.world.simulateMove(complex(x,y), move, False) ]))
            for x in self.world.N]
            for y in self.world.M]

    def diff(self, another):
        return max([max([abs(another[y][x] - self.U[y][x]) for x in range(self.N)]) for y in range(self.M)])

    def printable(self):
        return self.U

class QLearningAgent:
    '''Agent using Q-learning algorithm'''
    def __init__(self, d, world):
        s = world.board.findStarting()
        if(s == None):
            raise ValueError("Bad board")
        self.x, self.y = s
        self.reward = world.board[self.y][self.x]
        self.Q = {}
        self.N = {}
        self.d = d
        self.action = None

    @staticmethod
    def timeDiffPar(v):
        return 1/v

    def ExploreExploit(self, a):
        if(self.N[(a,self.x,self.y)] < Ne):
            return INFINITY
        else:
            return self.Q[(a, self.x, self.y)]

    def next(self):
        if(self.reward == None):
            self.x, self.y = world.board.findStarting()
            self.reward = world.board[self.y][self.x]

        if(not (self.action, self.x, self.y) in self.Q):
            self.Q[(self.action, self.x, self.y)] = 0
        if(not (self.action, self.x, self.y) in self.N):
            self.N[(self.action, self.x, self.y)] = 1

        if(self.world.board.at(complex(self.x, self.y)).isTerminal()):
                self.Q[(None, self.x, self.y)] = self.reward
                self.reward = None
                return

        self.Q[(self.action, self.x, self.y)] += self.timeDiffPar(self.N[(self.action, self.x, self.y)]) * \
            (self.reward - self.Q[(self.action, self.x, self.y)] + d * max([self.moveUtility(a) for a in allMoves]))

        possibilites = [ self.ExploreExploit(a) for a in allMoves ] 
        val = max(possibilites)
        self.action = possibilites.index(val)
        s = self.world.simulateMove(complex(self.x,self.y), self.action)
        self.x, self.y = (s.real, s.imag)
        self.reward = self.world.board.at(self.x,self.y).value

    def moveUtility(self, a):
        s,v,p = self.world.simulateMove(complex(self.x, self.y), a, False)[0]
        x,y = (s.real, s.imag)
        if(not (a, x, y) in self.Q):
            self.Q[(a,x,y)] = 0
        return self.Q[(a,x,y)]


    def printable(self):
        return [[max([self.Q[(x,y,a)] for a in allMoves]) for x in self.world.N] for y in self.world.M]

    def diff(self, another):
        U = self.printable()
        return max([max([abs(another[y][x] - U[y][x]) for x in range(self.N)]) for y in range(self.M)])

def createWorld(args):
    of = open(args.of, 'x')
    of.write(reduce(add, [str(x)+"  " for x in [args.N, args.M, args.a, args.b, args.r, args.d]]))
    of.write('\n')
    for x in range(args.M):
        for y in range(args.N):
            of.write('_:   ')
        of.write('\n')
    print("saved to " + args.of)

def fromFile(string):
    lnes = string.splitlines()
    vrs = lnes[0].split()
    N=int(vrs[0])
    M=int(vrs[1])
    a=float(vrs[2])
    b=float(vrs[3])
    r=float(vrs[4])
    d=float(vrs[5])
    return (d, World(N,M,a,b,r,lnes[1:]))

def runQAgent(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read()) 
    agent = QLearningAgent(d, world)
    runAgent(agent)
        
def runVAgent(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read()) 
    agent = ValueIterateAgent(d, world)
    runAgent(agent)


mainParser = argparse.ArgumentParser(description="Create or run AI agents in virtual 2D worlds")
#mainParser.set_defaults(func = id)
subparses = mainParser.add_subparsers()
createWorldParsers = subparses.add_parser('create_world')
createWorldParsers.add_argument("N", type=int, help="width")
createWorldParsers.add_argument("M", type=int, help="height")
createWorldParsers.add_argument("a", type=float, help="chance to move according to wishes")
createWorldParsers.add_argument("b", type=float, help="chance to misstep")
createWorldParsers.add_argument("r", type=float, help="reward for normal tile exploration")
createWorldParsers.add_argument("d", type=float, help="discount value (lower provide faster algorithm but lower precision)")
createWorldParsers.add_argument("oF", help="where to save file with world description")

createWorldParsers.set_defaults(func=createWorld)
runQAgentParser = subparses.add_parser('run_qagent')
runQAgentParser.add_argument("iF", help="path from which to load world")
runQAgentParser.set_defaults(func=runQAgent)
runVAgentParser = subparses.add_parser('run_vagent')
runVAgentParser.add_argument("iF", help="path from which to load world")
runVAgentParser.set_defaults(func=runVAgent)

if(__name__ == "__main__"):
    args = mainParser.parse_args()
    args.func(args)

