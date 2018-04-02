import random
from collections import deque
from utils import MemTooSmallError

class Memory(object):
    def __init__(self, max_length):
        self.max_length = max_length
        self.memory = deque(maxlen=max_length)

    def store(self, transition):
        self.memory.append(transition)

    def random_sample(self, sample_size):
        try:
            return random.sample(self.memory, sample_size)
        except ValueError: # mem not filled enough to return batch
            raise MemTooSmallError
