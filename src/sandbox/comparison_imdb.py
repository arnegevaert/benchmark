from os import path
from models.scripts.imdb_cnn import CNN
from torchtext import data, datasets
import torch
import spacy
from captum.attr import TokenReferenceBase, configure_interpretable_embedding_layer

# source: https://captum.ai/tutorials/IMDB_TorchText_Interpret
nlp = spacy.load('en')


def predict_sentiment(model, sentence, min_len=5):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


if __name__ == '__main__':
    model_path = path.join(path.dirname(__file__), "../models/saved_models/imdb_cnn.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = path.join(path.dirname(__file__), "../../data/imdb_preproc")

    print(f"Getting IMDB dataset")
    text = data.Field()
    label = data.LabelField()
    fields = {"text": ("text", text), "label": ("label", label)}
    train_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train="train.json",
        test="test.json",
        format="json",
        fields=fields
    )
    """
    train_data, test_data = datasets.IMDB.splits(text, label, root=data_path)
    train_data, valid_data = train_data.split()

    print(f"Building vocab")
    max_vocab_size = 25000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_,
                     vectors_cache=path.join(data_path, "glove"))
    label.build_vocab(train_data)

    """
    print(f"Getting data iterators")
    batch_size = 64
    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=batch_size,
        device=device
    )

    input_dim = len(text.vocab)
    embedding_dim = 100
    n_filters = 100
    filter_sizes = [3, 4, 5]
    output_dim = 1
    dropout = 0.5
    pad_idx = text.vocab.stoi[text.pad_token]
    model = CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)
    model.load_state_dict(torch.load(model_path))

    token_reference = TokenReferenceBase(reference_token_idx=pad_idx)
    interpretable_emb = configure_interpretable_embedding_layer(model, "embedding")

    batch = next(train_iterator)
