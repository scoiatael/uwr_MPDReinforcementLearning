#!/bin/python
Ne = 10
INFINITY = 10000
vAgent_iter = 10
qAgent_iter = 800
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
    PointConversion = { "U":0-1j, "D":0+1j, "L":-1-0j, "R":1-0j }

    def __init__(self, state):
        if(state in Move.PointConversion.keys()):
            self.state=state
        elif abs(state - Move.PointConversion["U"]) < 0.1:
            self.state = "U"
        elif abs(state - Move.PointConversion["L"]) < 0.1:
            self.state = "L"
        elif abs(state - Move.PointConversion["R"]) < 0.1:
            self.state = "R"
        elif abs(state - Move.PointConversion["D"]) < 0.1:
            self.state = "D"
        else:
            raise ValueError("Bad move definition {}".format(str(state)))

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


    def processNeighbour(self, s, ns, p):
        counter = self.U[ns[1]][ns[0]] * p
        return counter

    def processTile(self, x, y):
        tile = self.world.board.at(complex(x,y))
        counter = tile.value
        if(tile.isTerminal()):
            return counter
        if(tile.isForbidden()):
            return 0
        maxs = self.expValues(x,y)
        counter += self.d * max(maxs)
        return counter

    def expValues(self,x,y):
        return [reduce(add, [
                    self.processNeighbour((x,y), (int(s.real), int(s.imag)), p) \
                            for s,v,p in self.world.simulateMove(complex(x,y), Move(move), False) ]) \
                                for move in allMoves]

    def bestMove(self,x,y):
        maxs = self.expValues(x,y)
        maxv = max(maxs)
        pos = maxs.index(maxv)
        return allMoves[pos]

    def next(self):
        self.U = [[0 for x in range(self.world.N)]] + [[0] + [self.processTile(x,y)
            for x in self.world.rangeN()] + [0]
            for y in self.world.rangeM()] + [[0 for x in range(self.world.N)]]

    def diff(self, another):
        return max([max([abs(another.U[y][x] - self.U[y][x]) for x in self.world.rangeN()]) for y in self.world.rangeM()])

    def printable(self):
        U = self.output()
        return reduce(add, reduce(add, [[" {: 9.2f} ".format(t) for t in ln] + ["\n"] for ln in U]))

    def output(self):
        return [lne[1:-1] for lne in self.U[1:-1]]

    def politics(self):
        return [[self.bestMove(x,y) for x in self.world.rangeN()] for y in self.world.rangeM()]

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

        self.reset()


    def reset(self):
        s = self.world.board.findStarting()
        if(s == None):
            raise ValueError("Bad board")
        self.x, self.y = s
        self.reward = self.world.board.at(complex(self.x,self.y)).value
        self.generateNewMove()

    def generateNewMove(self):
        possibilites = [ self.ExploreExploit(Move(a)) for a in allMoves ]
        val = max(possibilites)
        self.action = Move(allMoves[possibilites.index(val)])

    @staticmethod
    def timeDiffPar(v):
        return 1/(v+1)

    def ExploreExploit(self, a):
        if random() > EPSILON*4:
            return INFINITY*random()
        else:
            return self.Q[(a.state, self.x, self.y)]

    def next(self):
        s,v,p = self.world.simulateMove(complex(self.x,self.y), self.action)[0]
        nx, ny = (int(s.real), int(s.imag))
        nt = self.world.board.at(s)
        nreward = nt.value

        if(self.world.board.at(complex(self.x, self.y)).isTerminal()):
            for move in allMoves:
                self.Q[(move, self.x, self.y)] = self.reward
            self.reset()
            return


        self.N[(self.action.state, self.x, self.y)] += 1
        self.Q[(self.action.state, self.x, self.y)] += self.timeDiffPar(self.N[(self.action.state, self.x, self.y)]) * \
            (self.reward - self.Q[(self.action.state, self.x, self.y)] + \
             self.d * max([self.moveUtility(Move(a),nx,ny) for a in allMoves])) #+ ([self.Q[(None, nx, ny)]] if nt.isTerminal else [])))


        self.generateNewMove();
        self.x, self.y = (nx,ny)
        self.reward = nreward

    def moveUtility(self, a,x,y):
        return self.Q[(a.state,x,y)]

    def output(self):
        U = [[max([self.Q[(a,x,y)] for a in allMoves]) \
                for x in self.world.rangeN()] for y in self.world.rangeM()]
        return U

    def bestMove(self,x,y):
        maxs = [self.Q[(a,x,y)] for a in allMoves]
        maxv = max(maxs)
        pos = maxs.index(maxv)
        return allMoves[pos]

    def politics(self):
        return [[ self.bestMove(x,y)\
                for x in self.world.rangeN()] for y in self.world.rangeM()]

    def printable(self):
        U = self.output()
        return reduce(add, reduce(add, [[" {: 9.2f} ".format(t) for t in ln] + ["\n"] for ln in U])) + "\n at {} {} \n".format(self.x, self.y)

    def diff(self, another):
        U = self.output()
        an = another.output()
        return max([max([abs(an[y][x] - U[y][x]) for x in range(len(U[y]))]) for y in range(len(U))])

def createWorld(args):
    of = open(args.oF, 'w')
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
    return (d, World(N,M,a,b,r,lnes[1:M+1]))

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

def runPlotV(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read())
    iF.close()
    agentV = ValueIterateAgent(d, world)
    gatherPlotData(agentV, args.oF + '_vAgent', vAgent_iter)

def runPlotQ(args):
    iF = open(args.iF,'r')
    d, world = fromFile(iF.read())
    iF.close()
    agentQ = QLearningAgent(d, world)
    gatherPlotData(agentQ, args.oF + '_qAgent', qAgent_iter)

def gatherPlotData(a, oPath, num):
    oF = open(oPath + '.dat', 'w')
    s = 0
    c = 0
    while True:
        if c % 100 == 0 :
            print('.')
        for x in range(num):
            c+=1
            b = deepcopy(a)
            a.next()
            oF.write(reduce(add, reduce(add, [[ " {: 9.2f} ".format(x) for x in ln] for ln in a.output()])))
            s+= abs(a.diff(b))
            oF.write('\n')
        if(s < 0.2 or c > 10000):
            break
        else:
            s = 0
    oF.close()
    oFp= open(oPath + '.pol', 'w')
    poli = a.politics()
    for y in a.world.rangeM():
        for x in a.world.rangeN():
            if a.world.board.at(complex(x,y)).isForbidden():
                poli[y-1][x-1]="X"
            if a.world.board.at(complex(x,y)).isTerminal():
                if a.world.board.at(complex(x,y)).value < 0:
                    poli[y-1][x-1]="-"
                else:
                    poli[y-1][x-1]="+"
    oFp.write(reduce(add, [reduce(add, [ x + " " for x in ln] ) + "\n" for ln in poli]))
    oFp.close()

    


mainParser = argparse.ArgumentParser(description="Create or run AI agents in virtual 2D worlds")
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

plotParserV = subparses.add_parser('plot_data_vagent')
plotParserV.add_argument('iF', help="world path")
plotParserV.add_argument('oF', help='output path')
plotParserV.set_defaults(func=runPlotV)
plotParserQ = subparses.add_parser('plot_data_qagent')
plotParserQ.add_argument('iF', help="world path")
plotParserQ.add_argument('oF', help='output path')
plotParserQ.set_defaults(func=runPlotQ)

if(__name__ == "__main__"):
    args = mainParser.parse_args()
    args.func(args)

