# Simple class to pass the score to main function as attribute
import os


class Scorer:
    def __init__(self, is_distributed=False, name=None):
        self.score = None
        self.is_distributed = is_distributed
        self.fpath = os.path.join('./../saved', f'score_{name}.txt')
        if not os.path.exists(self.fpath):
            with open(self.fpath, 'w') as f:
                f.write('0.0')

    def set_score(self, score):
        if self.is_distributed:
            with open(self.fpath, 'w') as f:
                f.write(str(score))
        else:
            self.score = score

    def get_score(self):
        if self.is_distributed:
            with open(self.fpath, 'r') as f:
                score = float(f.read())
        else:
            score = self.score
        return score
