# Simple class to pass the score to main function as attribute
class Scorer:
    def __init__(self, is_distributed=False):
        self.score = None
        self.is_distributed = is_distributed

    def set_score(self, score):
        if self.is_distributed:
            with open('./../saved/score.txt', 'w') as f:
                f.write(str(score))
        else:
            self.score = score

    def get_score(self):
        if self.is_distributed:
            with open('./../saved/score.txt', 'r') as f:
                score = float(f.read())
        else:
            score = self.score
        return score
