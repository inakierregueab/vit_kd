# Simple class to pass the score to main function as attribute
class Scorer:
    def __init__(self):
        self.score = None

    def set_score(self, score):
        self.score = score
