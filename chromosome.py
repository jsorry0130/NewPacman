import numpy as np


class Chromosome():
    def __init__(self):
        self.fitness = 0

        hidden_node = 10
        self.w1 = np.random.randn(4, hidden_node)  # 4행 10열 행렬 생성
        self.w2 = np.random.randn(hidden_node, 20)
        self.w3 = np.random.randn(20, hidden_node)
        self.w4 = np.random.randn(hidden_node, 4)
        # 4, 10, 20, 10, 4 개의 노드가 연결되어있는 구조의 가중치(연결망) 생성

    def forward(self, inputs):  # 입력값을 토대로한 결과 산출 함수
        net = np.matmul(inputs, self.w1)
        net = self.relu(net)
        net = np.matmul(net, self.w2)
        net = self.relu(net)
        net = np.matmul(net, self.w3)
        net = self.relu(net)
        net = np.matmul(net, self.w4)
        # 입력 값과 다음 계층의 가중치 값들을 행렬 곱하는 것을 반복하여 출력값을 산출
        net = self.softmax(net)  # 각 값의 적합 확률을 의미하기 위해 정규화
        return net

    def relu(self, x):  # 0이하의 값들은 필요없으므로 0 이하는 0으로 만드는 함수
        return x * (x >= 0)

    def softmax(self, x):  # 모든 값을 0~1 사이로 정규화하여 모든 값들의 합이 1이 되게하는 함수
        return np.exp(x) / np.sum(np.exp(x), axis=0)
