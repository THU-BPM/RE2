from torch.utils.data import Dataset

class RelationDataset(Dataset):
    def __init__(self, texts, relations):
        self.texts = texts
        self.relations = relations

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        relation = self.relations[idx]
        return text, relation

