import random as rnd
from copy import deepcopy
import numpy as np



def crossover(N_CHILDREN, best_chromosomes):
    for i in range(N_CHILDREN):
        new_chromosome = deepcopy(best_chromosomes[0])
        a_chromosome = rnd.choice(best_chromosomes)  # 부모 chromosome 랜덤 선택
        b_chromosome = rnd.choice(best_chromosomes)
        # chromosome의 인공신경망 구조 중 가중치 부분(2차원배열, 행렬값)을 crossover
        # single crossover로 crossover 지점(cut)은 무작위
        cut = rnd.randint(0, new_chromosome.w1.shape[1])  # shape[1]은 열.
        new_chromosome.w1[i, :cut] = a_chromosome.w1[i, :cut]
        new_chromosome.w1[i, cut:] = b_chromosome.w1[i, cut:]

        cut = rnd.randint(0, new_chromosome.w2.shape[1])
        new_chromosome.w2[i, :cut] = a_chromosome.w2[i, :cut]
        new_chromosome.w2[i, cut:] = b_chromosome.w2[i, cut:]

        cut = rnd.randint(0, new_chromosome.w3.shape[1])
        new_chromosome.w3[i, :cut] = a_chromosome.w3[i, :cut]
        new_chromosome.w3[i, cut:] = b_chromosome.w3[i, cut:]

        cut = rnd.randint(0, new_chromosome.w4.shape[1])
        new_chromosome.w4[i, :cut] = a_chromosome.w4[i, :cut]
        new_chromosome.w4[i, cut:] = b_chromosome.w4[i, cut:]
        # i번째 자식은 i번째 노드(행)에 대한 가중치만 crossover
        best_chromosomes.append(new_chromosome)  # 상위 chromosome 목록에 교배 완료된 자식 chromosome 추가


def mutation(N_POPULATION, N_BEST, N_CHILDREN, best_chromosomes, PROB_MUTATION, chromosomes =[]):
    chromosomes.clear() # mutation 후 세대 교체가 이뤄지므로 그 전에 현재 세대 초기화

    # 총 8개의 상위 chromosome 목록에 mutation을 2번 수행하여 16개의 새로운 chromosome 탄생
    for i in range(int(N_POPULATION / (N_BEST + N_CHILDREN))):
        for bg in best_chromosomes:
            new_chromosome = deepcopy(bg)

            mean = 20  # 평균
            stddev = 10  # 표준 변차

            # 평균 20, 분산 10의 랜덤한 실수의 행렬을 생성하고 100으로 나누어 다시 랜덤한 정수의 행렬과 곱함
            # 각 계층의 가중치 값(행렬)에 대해 수행
            if rnd.uniform(0, 1) < PROB_MUTATION:  # PROM_MUTATION의 확률로 수행
                new_chromosome.w1 += new_chromosome.w1 * np.random.normal(mean, stddev,
                                                                  size=(4, 10)) / 100 * np.random.randint(-1, 2,
                                                                                                          (4, 10))
            if rnd.uniform(0, 1) < PROB_MUTATION:
                new_chromosome.w2 += new_chromosome.w2 * np.random.normal(mean, stddev,
                                                                  size=(10, 20)) / 100 * np.random.randint(-1, 2,
                                                                                                           (10, 20))
            if rnd.uniform(0, 1) < PROB_MUTATION:
                new_chromosome.w3 += new_chromosome.w3 * np.random.normal(mean, stddev,
                                                                  size=(20, 10)) / 100 * np.random.randint(-1, 2,
                                                                                                           (20, 10))
            if rnd.uniform(0, 1) < PROB_MUTATION:
                new_chromosome.w4 += new_chromosome.w4 * np.random.normal(mean, stddev,
                                                                  size=(10, 4)) / 100 * np.random.randint(-1, 2,
                                                                                                          (10, 4))
            chromosomes.append(new_chromosome)