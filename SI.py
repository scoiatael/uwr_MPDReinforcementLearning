#!/bin/python
Ne = 10
INFINITY = 10000
vAgent_iter = 10
qAgent_iter = 1000
EPSILON=0.05

import argparse
import readline
import cmd
import os.path
from random import random
from functools import reduce
from operator import *
from sys import exit
from copy import deepcopy

def runAgent(agent):
    c = cmd.Cmd()
    c.postcmd = (lambda x, y: print(agent.printable()))
    c.do_n = (lambda x: agent.next())
    c.do_q = (lambda x: exit(0))
    c.cmdloop()

class Tile:
    '''Single tile of world'''
    def __init__(self, t, v):
        state, value = t.split(':')
        self.state=state
        if(not type(v) is float):
            raise ValueError("Value for tile not provided");
        if(value != ""):
            self.value = float(value)
        else:
            self.value = v

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
        x = int(p.imag)
        y = int(p.real)
        assert x < len(self.board), "x too far: {} out of {}".format(x, len(self.board))
        assert y < len(self.board[x]), "y too far: {} out of {}".format(y, len(self.board[x]))

        return self.board[x][y]

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

allMoves = "U D R L".split()

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
        assert len(self.board.board[0]) == self.N, "bad board x size: {} vs {}".format(len(self.board.board[0]), self.N)
        assert len(self.board.board) == self.M, "bad board y size: {} vs {}".format(len(self.board.board), self.M)

    def simulateMove(self, state, move, sim = True):
        pAndR = [ (ns, self.board.at(ns)) for ns in [ state + p.toPoint() for p in move.possibleOutcomes()] ]
        pAndR = [ (state, self.board.at(state).value) if b.isForbidden() else (s,b.value) for s,b in pAndR ]
        pAndR = [ ( pAndR[0][0],pAndR[0][1], self.a ), ( pAndR[1][0], pAndR[1][1], self.b ), ( pAndR[2][0], pAndR[2][1], self.b ) ]
        if( not sim ):
            return pAndR
        r = random() - self.a
        if(r < 0):
            return pAndR
        if(r < self.b):
            return [pAndR[1], pAndR[0], pAndR[2]]
        return [pAndR[2], pAndR[0], pAndR[1]]

    def rangeN(self):
        return range(self.N)[1:-1]

    def rangeM(self):
        return range(self.M)[1:-1]


class ValueIterateAgent:
    '''Agent using value iteration algorithm'''
    def __init__(self, d, world):
        self.world = world
        self.d = d
        self.U = [[ 0 for x in range(world.N)] for y in range(world.M)]


    def processTile(self, x, y):
        tile = self.world.board.at(complex(x,y))
        counter = tile.value
        if(tile.isTerminal()):
            return counter
        counter += self.d * max([reduce(add, [
                    self.U[int(s.imag)][int(s.real)] * p for s,v,p in self.world.simulateMove(complex(x,y), Move(move), False) ]) \
                            for move in allMoves])
        return counter

    def next(self):
        self.U = [[0 for x in range(self.world.N)]] + [[0] + [self.processTile(x,y)
            for x in self.world.rangeN()] + [0]
            for y in self.world.rangeM()] + [[0 for x in range(self.world.N)]]

    def diff(self, another):
        return max([max([abs(another.U[y][x] - self.U[y][x]) for x in self.world.rangeN()]) for y in self.world.rangeM()])

    def printable(self):
        U = self.output()
        return reduce(add, reduce(add, [["{: 7.1f}".format(t) for t in ln] + ["\n"] for ln in U]))

    def output(self):
        return [lne[1:-1] for lne in self.U[1:-1]]

class QLearningAgent:
    '''Agent using Q-learning algorithm'''
    def __init__(self, d, world):
        self.world = world
        self.Q = {}
        self.N = {}
        for a in allMoves+[None]:
            for x in self.world.rangeN():
                for y in self.world.rangeM():
                    self.Q[(a,x,y)] = 0
                    self.N[(a,x,y)] = 0
        self.d = d

        s = world.board.findStarting()
        if(s == None):
            raise ValueError("Bad board")
        self.x, self.y = s
        self.reward = world.board.at(complex(self.x,self.y)).value

        possibilites = [ self.ExploreExploit(Move(a)) for a in allMoves ]
        val = max(possibilites)
        self.action = Move(allMoves[possibilites.index(val)])

    @staticmethod
    def timeDiffPar(v):
        return 1/(v+1)

    def ExploreExploit(self, a):
#        if(self.N[(a.state,self.x,self.y)] < Ne):
        if random() > EPSILON*4:
            return INFINITY*random()
        else:
            return self.Q[(a.state, self.x, self.y)]

    def next(self):
        s,v,p = self.world.simulateMove(complex(self.x,self.y), self.action)[0]
        nx, ny = (int(s.real), int(s.imag))
        nreward = self.world.board.at(s).value

        if(self.world.board.at(complex(self.x, self.y)).isTerminal()):
            self.Q[(None, self.x, self.y)] = self.reward

#            s = self.world.board.findStarting()
#            self.x, self.y = s
#            self.reward = self.world.board.at(complex(self.x,self.y)).value

#            possibilites = [ self.ExploreExploit(Move(a)) for a in allMoves ]
#            val = max(possibilites)
#            self.action = Move(allMoves[possibilites.index(val)])
#            return
#            self.reward = None
#        if(self.reward == None):
#            self.x, self.y = self.world.board.findStarting()
#            self.reward = self.world.board.at(complex(self.x, self.y)).value


        self.N[(self.action.state, self.x, self.y)] += 1
        self.Q[(self.action.state, self.x, self.y)] += self.timeDiffPar(self.N[(self.action.state, self.x, self.y)]) * \
            (self.reward - self.Q[(self.action.state, self.x, self.y)] + \
             self.d * max([self.moveUtility(Move(a),nx,ny) for a in allMoves] ))#+ [self.Q[(None, nx, ny)]]))

     #   print("new val: " + str(self.Q[(self.action, self.x, self.y)]))

        possibilites = [ self.ExploreExploit(Move(a)) for a in allMoves ]
        val = max(possibilites)
        self.action = Move(allMoves[possibilites.index(val)])
     #   print("new act: " + str(self.action.state))
        self.x, self.y = (nx,ny)
        self.reward = nreward
     #   print("new reward: " + str(nreward))

    def moveUtility(self, a,x,y):
        s,v,p = self.world.simulateMove(complex(x, y), a, False)[0]
        x,y = (s.real, s.imag)
        if(not (a.state, x, y) in self.Q):
            self.Q[(a.state,x,y)] = 0
        return self.Q[(a.state,x,y)]

    def output(self):
        U = [[max([self.Q[(a,x,y)] for a in allMoves] + [self.Q[(None, self.x, self.y)]]) 
            for x in self.world.rangeN()] for y in self.world.rangeM()]
        return U

    def printable(self):
        U = self.output()
        return reduce(add, reduce(add, [["{: 7.1f}".format(t) for t in ln] + ["\n"] for ln in U])) + "\n at {} {} \n".format(self.x, self.y)

    def diff(self, another):
        U = self.output()
        an = another.output()
        return max([max([abs(an[y][x] - U[y][x]) for x in range(len(U[y]))]) for y in range(len(U))])

def createWorld(args):
    of = open(args.oF, 'x')
    of.write(reduce(add, [str(x)+"  " for x in [args.N+2, args.M+2, args.a, args.b, args.r, args.d]]))
    of.write('\n')
    for y in range(args.N+2):
        of.write('F:   ')
    of.write('\n')
    for x in range(args.M):
        of.write('F:   ')
        for y in range(args.N):
            of.write('_:   ')
        of.write('F:   ')
        of.write('\n')
    for y in range(args.N+2):
        of.write('F:   ')
    of.write('\n')
    of.close()
    print("saved to " + args.oF)

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
    iF.close()
    agent = QLearningAgent(d, world)
    runAgent(agent)

def runVAgent(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read())
    iF.close()
    agent = ValueIterateAgent(d, world)
    runAgent(agent)

def runPlot(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read())
    iF.close()
    agentV = ValueIterateAgent(d, world)
    agentQ = QLearningAgent(d, world)
    gatherPlotData(agentV, args.oF + '_vAgent', vAgent_iter)
    gatherPlotData(agentQ, args.oF + '_qAgent', qAgent_iter)

def gatherPlotData(a, oPath, num):
    oF = open(oPath + '.dat', 'x')
    s = 0
    c = 0
    while True:
        for x in range(num):
            c+=1
            b = deepcopy(a)
            a.next()
            oF.write(reduce(add, reduce(add, [[ "{: 7.1f}".format(x) for x in ln] for ln in a.output()])))
            s+= abs(a.diff(b))
            oF.write('\n')
        if(s < 0.2 or c > 10000):
            break
        else:
            s = 0
    oF.close()
    


mainParser = argparse.ArgumentParser(description="Create or run AI agents in virtual 2D worlds")
#mainParser.set_defaults(func = id)
subparses = mainParser.add_subparsers()
createWorldParsers = subparses.add_parser('create_world')
createWorldParsers.add_argument("-N", type=int, help="width",required=True)
createWorldParsers.add_argument("-M", type=int, help="height",required=True)
createWorldParsers.add_argument("-a", type=float, help="chance to move according to wishes",required=True)
createWorldParsers.add_argument("-b", type=float, help="chance to misstep",required=True)
createWorldParsers.add_argument("-r", type=float, help="reward for normal tile exploration",required=True)
createWorldParsers.add_argument("-d", type=float, help="discount value (lower provide faster algorithm but lower precision)",required=True)
createWorldParsers.add_argument("oF", help="where to save file with world description")

createWorldParsers.set_defaults(func=createWorld)
runQAgentParser = subparses.add_parser('run_qagent')
runQAgentParser.add_argument("iF", help="path from which to load world")
runQAgentParser.set_defaults(func=runQAgent)
runVAgentParser = subparses.add_parser('run_vagent')
runVAgentParser.add_argument("iF", help="path from which to load world")
runVAgentParser.set_defaults(func=runVAgent)

plotParser = subparses.add_parser('plot_data')
plotParser.add_argument('iF', help="world path")
plotParser.add_argument('oF', help='output path')
plotParser.set_defaults(func=runPlot)

if(__name__ == "__main__"):
    args = mainParser.parse_args()
    args.func(args)

