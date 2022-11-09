class Accumulator:
    def __init__(self):
        self._dic = {}

    def append(self, dic):
        for k in dic:
            if k in self._dic:
                self._dic[k].append(dic[k])
            else:
                self._dic[k] = [dic[k]]

    def mean(self):
        return {k: sum(self._dic[k]) / len(self._dic[k]) for k in self._dic.keys()}
