if __name__ == '__main__':
    import pygame_functions as pgf
import random as rnd
import threading
import numpy as np
from copy import deepcopy  # 내부에 객체들까지 모두 새롭게 copy
from chromosome import Chromosome
import pygame
import time
import sys
import csv
import genetic_algorithm as ga

# 게임에 사용할 맵 구조 9는 벽, 1은 포인트가 있는 길을 의미
gamegrid = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
            9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
            9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
            9, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 9,
            9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9,
            9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9,
            9, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 9,
            9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9,
            9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9,
            9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
            9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

# 팩맨이 포인트를 먹을시 1이 0이 되므로 맵구조가 바뀜, 따라서 새로운 게임을
gamegrid_restart = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                    9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 0, 9, 9, 0, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 0, 0, 0, 0, 0, 0, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 9, 9, 9, 9, 9, 1, 9, 9, 0, 9, 9, 9, 9, 9, 9, 9, 9, 0, 9, 9, 1, 9, 9, 9, 9, 9, 9,
                    9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 1, 9,
                    9, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 9,
                    9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9,
                    9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 1, 9, 9, 9,
                    9, 1, 1, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 9, 9, 1, 1, 1, 1, 1, 1, 9,
                    9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9,
                    9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9,
                    9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

sys.setrecursionlimit(10000)  # 재귀함수 최대깊이 설정
#  Game = False

N_POPULATION = 16  # 한 모집단의 수, 즉 한 세대(한 주기)가 가지는 chromosome의 수
N_BEST = 4  # 높은 Fitness 순 대로 선정된 상위 chromosome
N_CHILDREN = 4  # crossover를 통해 새롭게 생성할 chromosome의 수
PROB_MUTATION = 0.2  # mutation이 발생할 확률

best_chromosomes = None  # 상위 chromosome을 담을 리스트 초기화
n_gen = 0  # 세대 차수 초기화
FPS = 500  # frame per second , 빠른 진화속도를 위해 높은 프레임


trikey = 0  # 메뉴 선택 변수
dlt = True
directions = {0: (1, 0), 1: (-1, 0), 2: (0, -1), 3: (0, 1)}  # 각 방향에 해당하는 좌표 변화 값
width, height = 672, 812
grid = 24
column = width // grid  # 맵 구조 열 개수
low = height // grid  # 맵 구조 행 개수

pgf.screenSize(width, height)
score = 0

main1 = pygame.image.load('Resource\Main_image1.png')
main2 = pygame.image.load('Resource\Main_image2.png')

nextAnimation = pgf.clock() + 100

clear = False


# 게임화면에 현재 점수를 표기하는 함수
def writeScore(count):
    global score
    pygame.draw.rect(pgf.screen, (0, 0, 0), [500, 760, 300, 100])
    score += count
    font = pygame.font.Font('Resource\\NanumGothic.ttf', 25)
    text = font.render('Score : ' + str(score), True, (255, 255, 255))
    pgf.screen.blit(text, (500, 760))


# 게임화면에 현재 세대 차수를 표기하는 함수
def writeGeneration(num):
    pygame.draw.rect(pgf.screen, (0, 0, 0), [30, 760, 300, 100])
    font = pygame.font.Font('Resource\\NanumGothic.ttf', 25)
    text = font.render('Generation : ' + str(num), True, (255, 255, 255))
    pgf.screen.blit(text, (30, 760))


# 맵구조의 인덱스를 좌표로 변환
def i2xy(i):  
    sp = i % column
    ze = i // column  # // 연산자는 나누기 결과값을 int 형으로 반환하는 것.
    return (sp * grid) + (grid // 2), (ze * grid) + (grid // 2)  # ex) 50번째 인덱스는 위의 연산을 거쳐 539, 35 라는 좌표가 생성된다


# 좌표를 맵구조의 인덱스로 변환
def xy2i(x, y): 
    sp, ze = (x - grid // 2) // grid, (y - grid // 2) // grid
    return ze * column + sp


# 맵에 배치할 포인트 클래스
class Point:
    def __init__(self, pos, imageFile):
        self.x, self.y = pos  # pos 파라미터는 2개의 값을 받아야한다. ex) i2xy()
        self.sprite = pgf.makeSprite(imageFile)  # 포인트 이미지를 받아 스프라이트로 저장
        pgf.moveSprite(self.sprite, self.x, self.y, centre=True)  # x, y의 좌표로 배치


# 팩맨, 고스트 등의 움직이는 개체를 생성할 클래스
class Figure:
    def __init__(self, name, pos):
        self.name = name
        self.x, self.y = pos
        self.direction = 0  # 진행 방향 초기화
        self.rx, self.ry = 1, 0  # 진행 방향으로 진행하기위해 더해야할 좌표 초기화
        self.mode = 'hunt'  # 동작 초기화
        self.imageNo = 0
        self.ongrid = False
        self.i = 355

    def display(self):  # 스프라이트를 맵에 배치, 표기하는 함수
        pgf.moveSprite(self.sprite, self.x, self.y, centre=True)  # 좌표위치에 스프라이트를 배치
        pgf.showSprite(self.sprite)  # 배치된 스프라이트 표기

    def move(self):  # 좌표 변화를 통해 개체를 움직이는 함수
        self.x += self.rx
        self.y += self.ry
        self.i = xy2i(self.x, self.y)  # 변화된 좌표를 인덱스로 변환하여 저장
        x2, y2 = i2xy(self.i)  # 검증을 위해 다시 좌표로 변환하여 따로 저장
        self.ongrid = self.x == x2 and self.y == y2  # 좌표 검증

    def warp(self):  # 워프 지역에 닿으면 이어진 좌표로 워프시킴
        if gamegrid[self.i] not in (5, 6): return False
        self.i = self.i + 27 if gamegrid[self.i] == 5 else self.i - 27
        self.x, self.y = i2xy(self.i)
        return True

    def directionValid(self, direction, j):  # 진행 방향을 확인하는 함수
        rx, ry = directions[direction]  # directions 딕셔너리는 각 방향에 해당하는 변화 좌표값 2개를 가짐
        i = j + rx + ry * column  # 변화될 좌표에 해당하는 인덱스 계산 시 가로값은 그냥 더하고 세로값은 열의 수를 곱하여 더함
        # self.x += self.rx
        # self.y += self.ry
        # i = xy2i(self.x, self.y)
        return gamegrid[i] != 9  # 맵 구조상 값이 9인 곳은 벽이므로 벽이 아닌곳은 진행 방향으로 설정 가능

    def changemode(self, mode): # 상태를 전환하는 함수
        pgf.hideSprite(self.sprite)  # 현재 스프라이트 이미지를 숨김
        self.mode = mode  # 상태 전환
        self.sprite = self.sprites[mode][0]  # 파라미터로 받은 mode에 해당하는 값의 이미지로 스프라이트 변환
        self.imageNo = 0

    def changedirection(self, direction):  # 진행 방향 전환
        if self.directionValid(direction, self.i):  # 방향확인함수로 확인
            self.direction = direction
            self.rx, self.ry = directions[direction]
            return True

    def animate(self):  # 애니메이션, 즉 개체들이 스프라이트 연속 교체를 통해 움직이게하는 함수
        sprite, animationimages, directionsdependent = self.sprites[self.mode]
        self.imageNo = (self.imageNo + 1) % animationimages
        if directionsdependent:
            self.imageNo += animationimages * self.direction
        pgf.changeSpriteImage(sprite, self.imageNo)


# 시뮬레이션(진화 학습)에 쓰이는 팩맨을 생성하는 클래스
class Pacman(Figure):
    def __init__(self, name, pos, chromosome):
        Figure.__init__(self, name, pos)
        self.sprites = {'hunt': [pgf.makeSprite('Resource\Teil_17_Pacman_Tileset.png', 12), 3, True],
                        'dead': [pgf.makeSprite('Resource\Teil_17_pacman_die.png', 12), 12, False]}
        self.sprite = self.sprites[self.mode][0]  # 팩맨의 초기모습 스프라이트 초기화(hunt mode)
        self.keyboardmemory = 0
        self.chromosome = chromosome
        self.timer = 0
        self.last_eat_time = 0
        self.fitness = 0 # 적합도
        self.evasion = 0 # 유령 회피 성공 수

    def motinlogic(self):  # 팩맨의 행동 로직
        if not self.ongrid: return
        #if self.warp(): return
        self.pointEat()
        break_switch = 0  # 근처에 유령이 있다면 1로 변경되는 스위치
        while True:  # 유령이 주변에 없을 때 행동
            while True:
                newdirection = rnd.randrange(4)  # 랜덤으로 방향을 설정
                if self.direction == 0 and newdirection != 1 or \
                        self.direction == 1 and newdirection != 0 or \
                        self.direction == 2 and newdirection != 3 or \
                        self.direction == 3 and newdirection != 2:
                    break

            for i in self.ghost_sensor():  # 유령 감지 센서의 값 확인
                if i != 0:  # 유령이 감지되었다면
                    break_switch = 1  # 스위치 설정

            if self.changedirection(newdirection) or break_switch == 1:
                break

        if break_switch == 1:  # 유령이 근처에 있으므로 학습된 chromosome 에게 방향 판단을 맡김
            newdirection = self.dir_decision(chromosome)  # chromosome이 결정한 방향으로 갱신
            self.changedirection(newdirection)
            if not self.directionValid(self.direction, self.i):  # 진행 불가 방향이면 움직이지 않음
                self.rx, self.ry = 0, 0

    def pointEat(self):  # 포인트를 먹는 함수
        if gamegrid[self.i] not in (1, 2):
            return  # 팩맨이 위치한 곳에 포인트가 없다면 실행 X

        elif gamegrid[self.i] == 1:
            writeScore(10)
            self.last_eat_time = self.timer  # 마지막으로 먹은 시간 갱신
        gamegrid[self.i] = 0   # gamegrid(맵구조) 에서 먹힌 포인트는 0으로
        pgf.killSprite(point_d[self.i].sprite)  # 자리에있던 포인트 스프라이트 제거.
        del point_d[self.i]

    def ghost_sensor(self):  # 유령이 팩맨 주변에 있는지 감지하는 센서

        pacman_pos = xy2i(self.x, self.y)  # 팩맨과 고스트들의 현재 위치 인덱스 저장
        blinky_pos = xy2i(blinky.x, blinky.y)
        pinky_pos = xy2i(pinky.x, pinky.y)
        inky_pos = xy2i(inky.x, inky.y)
        clyde_pos = xy2i(clyde.x, clyde.y)

        ghost_list = [blinky_pos, pinky_pos, inky_pos, clyde_pos]  # 고스트들의 현재 위치가 담긴 리스트

        sensor = [ [ [], [], [], [] ],
                   [ [], [], [], [] ],
                   [ [], [], [], [] ],
                   [ [], [], [], [] ] ]  # 2번째 교차로 까지 탐색할 4방향 센서를 담을 리스트

        search_func(self, sensor, pacman_pos, pacman_pos, 2)  # 맵의 길을 탐색하는 함수

        ghost_input = [0, 0, 0, 0]  # 고스트가 4방향중 어디있는지 알려줄 input값

        for ghost in ghost_list:
            for i, s in enumerate(sensor):
                if ghost in s:  # 먼저 각 센서의 뿌리에 있는지 확인
                    ghost_input[i] = 1
                else:  # 아니라면
                    for j, a in enumerate(s):  # 뿌리가 가지고 있는 각 가지에 있는지 확인
                        if j > 3: break  # 4번째 가지를 지나면 뿌리가 나오므로 뿌리는 패스
                        if ghost in a:  # 가지안에 있다면
                            ghost_input[i] = 1

        # print(ghost_input)
        return ghost_input

    def dir_decision(self, chromosome):  # chromosome에 따라 진행 방향을 결정하는 함수
        self.chromosome = chromosome
        input = self.ghost_sensor()  # chromosome이 진행방향을 결정하기 위해 인식해야할 현재 상황(팩맨 주변 유령)
        outputs = chromosome.forward(input)  # chromosome이 현재 상황(input)을 인식하여 판단한 결과 값
        # 각 값은 각 방향에 대한 합당한 확률을 의미

        for i in range(4):  # 현재 진행 불가한 방향은 outputs에서 제외
         if not self.directionValid(i, self.i):
            outputs[i] = 0

        outputs = np.argmax(outputs)  # 각장 높은 확률을 가지는 방향으로 갱신

        #  fitness(적합도)로 쓰일 유령 회피 성공 수 계산
        if input[outputs] == 0:  # chromosome 이 결정한 방향에 유령이 없다면 evasion(회피) 변수 값 증가
            self.evasion += 1
        elif input[outputs] == 1:
            self.evasion -= 2

        return outputs

# 팩맨 게임을 실행하는 함수
def run():
    global clear
    global nextAnimation

    while True:

        pgf.tick(FPS)  # 프레임 속도
        if pgf.keyPressed('right'): pacman.keyboardmemory = 0
        if pgf.keyPressed('left'): pacman.keyboardmemory = 1
        if pgf.keyPressed('up'): pacman.keyboardmemory = 2
        if pgf.keyPressed('down'): pacman.keyboardmemory = 3

        for Figure in Figures:
            if pgf.clock() > nextAnimation:
                Figure.animate()
            Figure.motinlogic()  # 모션 로직대로 행동
            Figure.move()  # 움직임
            if Figure.name != 'pacman':  # 고스트가 팩맨에게 부딪힐 때
                if Figure.i == pacman.i:
                    if Figure.mode == ('hunt'): # 고스트가 hunt 상태라면
                        pacman.changemode('dead')  # 팩맨 사망
                        clear = True  
                        break
                    if Figure.mode in ('frighten', 'blink'):  # 고스트가 도망가는 상태라면
                        Figure.changemode('dead')  # 고스트가 사망
                        writeScore(100) 
            Figure.display()
        if clear == True:
            clear = False
            break

        if pgf.clock() > nextAnimation:
            nextAnimation += 100

        pgf.updateDisplay()

        if pgf.keyPressed('esc'):
            break


# 맵 구조에서 길을 탐색하는 함수
def search_func(self, sensor, pacman_pos, pos, n):
    # self : 탐색할 기준이 될 객체, sensor: 탐색 결과를 저장할 리스트
    # pacman_pos : 탐색 기점이 될 팩맨의 위치, pos : 탐색 기점이 될 위치
    if n == 0:  # 원하는 만큼 반복 후 종료 (무한 재귀 탈출)
        return 0

    for i, v in enumerate(sensor):
        if i == 4: break  # 4방향이므로 4번까지만
        search_pos = pos  # 탐색 기점에서 출발
        s_direction = i  # 탐색 방향 설정
        rx, ry = directions[s_direction]  # 현재 방향의 좌표 변화값 저장
        if gamegrid[pos + rx + ry * column] == 9: continue  # 현재 방향대로 한칸 나아갔을 때 벽이라면 패스
        
        while True:  # 탐색 시작
            rx, ry = directions[s_direction]  # 탐색마다 초기화
            search_pos = search_pos + rx + ry * column  # 진행방향 대로 탐색위치 전진
            if search_pos == pacman_pos : break   # 센서 탐색 범위가 팩맨 위치와 겹친다면 그 이상 탐색할 이유가 없으므로 break
            if gamegrid[search_pos] == 9: break  # 현재탐색위치가 벽이라면 while 루프 탈출
            if not crossway_valid(search_pos):  # 교차로가 아니면 탐색 위치값 센서에 삽입
                v.append(search_pos)
                
                if not Figure.directionValid(self, s_direction, search_pos):  # 전진 방향 다음 칸이 막혔다면
                    for j in range(0, 4):  # 방향 전환(커브길)
                        if j == s_direction or j + s_direction in (1, 5):
                            continue  # 되돌아가는 방향은 고려하지 않음
                        if Figure.directionValid(self, j, search_pos):
                            s_direction = j  # 탐색방향 재설정
                            break
            # 교차로 탐색 시 교차로를 기점으로 다시 탐색하기 위해 재귀호출
            elif crossway_valid(search_pos):
                v.append(search_pos)
                search_func(self, v, pacman_pos, search_pos, n - 1)
                break


# 교차로 판단 함수
def crossway_valid(i):
    cross_bool = False
    way_num = 0  # 현 위치에서 진행 가능한 방향의 수

    if gamegrid[i + 1] in range(0, 3):  # 오른쪽에 길이 있다면
        way_num += 1
    if gamegrid[i - 1] in range(0, 3):  # 왼쪽에  길이 있다면
        way_num += 1
    if gamegrid[i + column] in range(0, 3):  # 아래쪽에 길이 있다면
        way_num += 1
    if gamegrid[i - column] in range(0, 3):  # 위쪽에 길이 있다면
        way_num += 1

    if way_num >= 3:  # 전환 가능 방향이 3개 이상이면 교차로
        cross_bool = True

    if gamegrid[i] == 9:  # 벽
        cross_bool = False

    return cross_bool

# 유령(blinky 제외) 생성 클래스
class ghost(Figure):
    def __init__(self, name, pos, imageFile):
        Figure.__init__(self, name, pos)
        self.sprites = {'hunt': [pgf.makeSprite(imageFile, 8), 2, True],
                        'frighten': [pgf.makeSprite('Resource\Teil_17_Ghost_frighten.png', 2), 2, False],
                        'blink': [pgf.makeSprite('Resource\Teil_17_Ghost_blink.png', 4), 4, False],
                        'dead': [pgf.makeSprite('Resource\Teil_17_Ghost_die.png', 4), 1, True]}
        self.sprite = self.sprites[self.mode][0]

    def motinlogic(self):
        if not self.ongrid: return
        #if self.warp(): return
        while True:  # 무작위로 이동
            while True:
                newdirection = rnd.randrange(4)
                if self.direction == 0 and newdirection != 1 or \
                        self.direction == 1 and newdirection != 0 or \
                        self.direction == 2 and newdirection != 3 or \
                        self.direction == 3 and newdirection != 2:
                    break
            if self.changedirection(newdirection):
                break


# 팩맨을 항상 추적하는 유령 Blinky를 생성하는 클래스
class Blinky(Figure):
    def __init__(self, name, pos, imageFile):
        Figure.__init__(self, name, pos)
        self.sprites = {'hunt': [pgf.makeSprite(imageFile, 8), 2, True],
                        'frighten': [pgf.makeSprite('Resource\Teil_17_Ghost_frighten.png', 2), 2, False],
                        'blink': [pgf.makeSprite('Resource\Teil_17_Ghost_blink.png', 4), 4, False],
                        'dead': [pgf.makeSprite('Resource\Teil_17_Ghost_die.png', 4), 1, True]}
        self.sprite = self.sprites[self.mode][0]
        self.keyboardmemory = 0

    def motinlogic(self):

        if not self.ongrid: return
        #if self.warp(): return

        while True:  # 기존 유령과 달리 무작위로만 움직이지 않고 계속해서 팩맨을 추적
            while True:
                tracex = pacman.x - blinky.x
                tracey = pacman.y - blinky.y
                newdirection = rnd.randrange(4)

                if self.direction == 0 and newdirection != 1 or \
                        self.direction == 1 and newdirection != 0 or \
                        self.direction == 2 and newdirection != 3 or \
                        self.direction == 3 and newdirection != 2:
                    break

                if (abs(tracex) > abs(tracey)):
                    if (tracex > 0):
                        newdirection = 0
                        break
                    else:
                        newdirection = 1
                        break

                elif (abs(tracey) > abs(tracex)):
                    if (tracey < 0):
                        newdirection = 2
                        break
                    else:
                        newdirection = 3
                        break
                if self.direction == 0 and newdirection != 1 or \
                        self.direction == 1 and newdirection != 0 or \
                        self.direction == 2 and newdirection != 3 or \
                        self.direction == 3 and newdirection != 2:
                    break

            if self.changedirection(newdirection):
                break


#  플레이어블 캐릭터 팩맨을 생성하는 클래스
class Gacman(Figure):
    def __init__(self, name, pos):
        Figure.__init__(self, name, pos)
        self.sprites = {'hunt': [pgf.makeSprite('Resource\Teil_17_Pacman_Tileset.png', 12), 3, True],
                        'dead': [pgf.makeSprite('Resource\Teil_17_pacman_die.png', 12), 12, False]}
        self.sprite = self.sprites[self.mode][0]
        self.keyboardmemory = 0

    def pointEat(self):
            if gamegrid[self.i] not in (1, 2): return
            gamegrid[self.i] = 0
            pgf.killSprite(point_d[self.i].sprite)
            writeScore(10)
            del point_d[self.i]

    def motinlogic(self):
            if not self.ongrid: return
            #if self.warp(): return
            self.pointEat()
            self.changedirection(self.keyboardmemory)  # 키보드입력대로 방향전환
            if not self.directionValid(self.direction, self.i):
                self.rx, self.ry = 0, 0


#  포인트 이미지 스프라이트를 배치해주는 함수
def pointSet(dlt):

    if dlt:  # 새 게임 시작시 맵구조를 복원
        point_d = {}
        for i, number in enumerate(gamegrid):
            gamegrid[i] = gamegrid_restart[i]

    else:
        point_d = {}
        for i, number in enumerate(gamegrid_restart):
            if number not in (1, 2): continue
            point_d[i] = Point(i2xy(i), 'Resource\Teil_17_Punkt.png') if number == 1 else Point(i2xy(i),
                                                                                                 'Resource\Teil_17_Punkt_gross.png')
            pgf.showSprite(point_d[i].sprite)

    return point_d


# 유령의 상태를 바꿔주는 함수
def changemodeGhosts(mode):
    for Figure in Figures:
        if Figure.name == 'pacman': continue
        if Figure.mode != 'dead':
            Figure.changemode(mode)
    if mode == 'frighten':
        timer1 = threading.Timer(5, changemodeGhosts, ('blink',)).start()
        timer2 = threading.Timer(8, changemodeGhosts, ('hunt',)).start()


# 메인 화면(시작화면)을 구성하는 함수
def main_UI():
    #global Game
    global trikey
    pgf.hideAll()
    pgf.updateDisplay()

    while True:
        #pgf.tick(50)
        pgf.screen.blit(main1, (183, 200))
        pgf.screen.blit(main2, (180, 310))
        pgf.updateDisplay()

        font = pygame.font.Font('Resource\\NanumGothic.ttf', 35)

        gamestart = font.render('G a m e   S t a r t', True, (255, 255, 255))
        pgf.screen.blit(gamestart, (180, 500))
        if trikey == 0:
            tri = Point((120, 520), 'Resource\Triangle.png')
            pgf.showSprite(tri.sprite)

        simulation = font.render('S i m u l a t i o n', True, (255, 255, 255))
        pgf.screen.blit(simulation, (180, 570))
        if trikey == 1:
            tri = Point((120, 590), 'Resource\Triangle.png')
            pgf.showSprite(tri.sprite)

        exit = font.render('E X I T', True, (255, 255, 255))
        pgf.screen.blit(exit, (180, 640))
        if trikey == 2:
            tri = Point((120, 660), 'Resource\Triangle.png')
            pgf.showSprite(tri.sprite)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    if (trikey < 2):
                        trikey += 1

                elif event.key == pygame.K_UP:
                    if (trikey > 0):
                        trikey -= 1

                pgf.hideAll()

        if pgf.keyPressed('enter'):
            if (trikey == 2):  # 종료
                pygame.quit()
                sys.exit()
                break
            else :
                break


if __name__ == "__main__":
    while True:
        pgf.screenSize(width, height)
        main_UI()
        if trikey == 0:  # 직접 플레이
            f = open('PlayerOutput.csv', 'w', newline='')  # 데이터 기록을 위한 csv 파일 생성
            wr = csv.writer(f)
            wr.writerow(["Time", "Score"])
            score = 0
            pgf.hideAll()
            pgf.screenSize(width, height)
            point_d = pointSet(True)  # 새 게임 시작전 맵 구조 복원
            pgf.setBackgroundImage('Resource\Teil_17_Spielfeld.png')
            pgf.setAutoUpdate(False)

            # 화면에 배치될 포인트, 유령, 팩맨 개체 생성
            point_d = pointSet(False)
            blinky = Blinky('blinky', (270, 420), 'Resource\Teil_17_Blinky_tileset.png')
            pinky = ghost('pinky', (390, 420), 'Resource\Teil_17_pinky_tileset.png')
            inky = ghost('inky', (360, 276), 'Resource\Teil_17_inky_tileset.png')
            clyde = ghost('clyde', (360, 276), 'Resource\Teil_17_clyde_tileset.png')
            pacman = Gacman('pacman', (336, 564))
            
            pgf.updateDisplay()
            Figures = [pacman, blinky, pinky, inky, clyde]
            pgf.setAutoUpdate(False)
            nextAnimation = pgf.clock() + 100
            StartT = time.time()
            run()
            EndT = round(time.time() - StartT, 2)
            wr.writerow([EndT, score])

        elif trikey == 1:  # 시뮬레이션
            chromosomes = [Chromosome() for _ in range(N_POPULATION)]  # 최초 세대 초기화
            f = open('output.csv', 'w', newline='')
            wr = csv.writer(f)
            wr.writerow(["Generation", "Genomes", "Evasion", "Time", "Score"])
            while True:
                n_gen += 1  # 세대 차수 기록
                if pgf.keyPressed('esc'):
                    break

                for i, chromosome in enumerate(chromosomes):
                    if pgf.keyPressed('esc'):
                        break
                    StartT = time.time()  # 시간 기록
                    score = 0
                    pgf.hideAll()
                    pgf.screenSize(width, height)
                    point_d = pointSet(True)  # 맵 구조 복원
                    writeGeneration(n_gen)
                    pgf.setBackgroundImage('Resource\Teil_17_Spielfeld.png')
                    pgf.setAutoUpdate(False)
                    
                    # 화면에 배치될 포인트, 유령, 팩맨 개체 생성
                    point_d = pointSet(False)
                    pacman = Pacman('pacman', (336, 564), chromosome=chromosome)  # 2번째 파라미터는 좌표
                    blinky = Blinky('blinky', (270, 420), 'Resource\Teil_17_Blinky_tileset.png')
                    pinky = ghost('pinky', (390, 420), 'Resource\Teil_17_pinky_tileset.png')
                    inky = ghost('inky', (360, 276), 'Resource\Teil_17_inky_tileset.png')
                    clyde = ghost('clyde', (360, 276), 'Resource\Teil_17_clyde_tileset.png')

                    pgf.updateDisplay()
                    Figures = [pacman, blinky, pinky, inky, clyde]
                    pgf.setAutoUpdate(False)
                    nextAnimation = pgf.clock() + 100
                    run()
                    EndT = round(time.time() - StartT,2)  # 시간 측정 끝

                    chromosome.fitness = pacman.evasion  # 적합도는 유령 회피 성공 수
                    wr.writerow([n_gen, i, pacman.evasion, EndT, score])  # csv 파일에 게임 결과 기록

                    if pgf.keyPressed('esc'):
                        n_gen = 0
                        break

                if pgf.keyPressed('esc'):
                    break

                # 유전 알고리즘을 이용한 진화 시작
                if best_chromosomes is not None:  # 퇴화 방지를 위해 이전 세대 상위 chromosome 까지 포함
                    chromosomes.extend(best_chromosomes)

                chromosomes.sort(key=lambda x: x.fitness, reverse=True)  # 적합도를 기준으로 내림차순 정렬

                print('===== Generaton #%s\tBest Fitness %s =====' % (n_gen, chromosomes[0].fitness))  # 세대 끝, 최고 적합도 출력

                best_chromosomes = deepcopy(chromosomes[:N_BEST])  # 상위 chromosome 4개 따로 저장

                ga.crossover(N_CHILDREN, best_chromosomes)  # crossover
                ga.mutation(N_POPULATION, N_BEST, N_CHILDREN, best_chromosomes, PROB_MUTATION, chromosomes)  #mutation

                if pgf.keyPressed('esc'):
                    break