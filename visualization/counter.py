class CorpusCounter():

    #TODO or not :<

    def __init__(self, dataset_path):

        self.qa_pairs = []
        self._load_dataset(dataset_path)

    def _load_dataset(self, dataset_path):

        with open(dataset_path, 'r', encoding='utf-8') as dataset:
            for line in dataset:
                line = line.strip('\n')
