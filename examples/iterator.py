class MyIterator:

    def __init__(self, data, labels, batch_size = 64):
        if len(data) != len(labels):
            raise ValueError("Length of data doesn't match length of labels.")
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.index = 0

    def __next__(self):
        x = self.data[self.index : self.index+self.batch_size]
        y = self.labels[self.index : self.index+self.batch_size]
        self.index += self.batch_size
        if self.index + self.batch_size >= len(self.labels):
            self.index = 0

        return (x,y)