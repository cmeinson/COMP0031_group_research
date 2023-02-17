
class Data:
    def __init__(self) -> None:
        # does reading and cleaning go here or do we add extra functions for that?
        raise NotImplementedError

    def transform(self):
        # will probably rename later. but something for merging attributes into binary ones?
        raise NotImplementedError

    def get_train_data(self):
        raise NotImplementedError

    def get_test_data(self):
        raise NotImplementedError

