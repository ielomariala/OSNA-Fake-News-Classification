from bert_dataloader import DataLoader_Bert
from bert_model import BertModel

def main():
    dloader = DataLoader_Bert('data')
    train_gen, valid_gen = dloader.prepare_training()
    
    model = BertModel()
    model.create_model()
    model.fit(train_gen, valid_gen)
    model.save_model("save_model")

    


if __name__ == '__main__':
    main()