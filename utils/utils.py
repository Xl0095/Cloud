class Accumulator:
    def __init__(self):
        self.v1 = []
        self.v2 = []

    def append(self, v1, v2):
        self.v1.append(v1)
        self.v2.append(v2)

    def mean(self):
        return sum(self.v1) / len(self.v1), sum(self.v2) / len(self.v2)
