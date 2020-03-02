import torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from os import path
import time
import spacy

# Source: https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convlayers = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
            ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # Text: [batch_size, sentence_length]
        embedded = self.embedding(text)  # [batch_size, sentence_length, embedding_dim]
        embedded = embedded.unsqueeze(1)  # [batch_size, 1, sentence_length, embedding_dim]
        # conved_n = [batch_size, n_filters, sent_len - filter_sizes[n]+1]
        conved = [F.relu(layer(embedded).squeeze(3)) for layer in self.convlayers]
        # pooled_n = [batch_size, n_filters]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, n_filters * len(filter_sizes)]
        return self.fc(cat)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def train(model, iterator, optimizer, loss):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for i, batch in enumerate(iterator):
        if (i+1)%25 == 0:
            print(f"Batch {i+1}/{len(iterator)}")
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        l = loss(predictions, 1 - batch.label)  # labels are parsed as 'pos' = 0, 'neg' = 1 => flip
        acc = binary_accuracy(predictions, batch.label)
        l.backward()
        optimizer.step()
        epoch_loss += l.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, loss):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            l = loss(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            epoch_loss += l.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def build_and_train(save=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text, label, train_iterator, valid_iterator, test_iterator = get_dataset()

    input_dim = len(text.vocab)
    embedding_dim = 100
    n_filters = 100
    filter_sizes = [3, 4, 5]
    output_dim = 1
    dropout = 0.5
    pad_idx = text.vocab.stoi[text.pad_token]
    n_epochs = 5
    save_location = path.join(path.dirname(__file__), "../saved_models")

    print(f"Building model")
    model = CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)

    print(f"Setting pretrained embeddings")
    pretrained_embeddings = text.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_embeddings)
    unk_idx = text.vocab.stoi[text.unk_token]
    model.embedding.weight.data[unk_idx] = torch.zeros(embedding_dim)
    model.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

    optimizer = optim.Adam(model.parameters())
    loss = nn.BCEWithLogitsLoss()
    model = model.to(device)
    loss = loss.to(device)

    best_valid_loss = float('inf')

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, loss)
        valid_loss, valid_acc = evaluate(model, valid_iterator, loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss and save:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), path.join(save_location, 'imdb_cnn.pt'))

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
    return model, text, label


def get_dataset():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = path.join(path.dirname(__file__), "../../../data")
    print(f"Getting IMDB dataset")
    text = data.Field(batch_first=True)
    label = data.LabelField(dtype=torch.float)
    fields = {"text": ("text", text), "label": ("label", label)}
    train_data, valid_data, test_data = data.TabularDataset.splits(
        path=path.join(data_path, "imdb_preproc"),
        train="train.json",
        test="test.json",
        validation="valid.json",
        format="json",
        fields=fields
    )
    print(f"Building vocab")
    max_vocab_size = 25000
    text.build_vocab(train_data,
                     max_size=max_vocab_size,
                     vectors="glove.6B.100d",
                     unk_init=torch.Tensor.normal_,
                     vectors_cache=path.join(data_path, "glove"))
    label.build_vocab(train_data)

    print(f"Getting data iterators")
    batch_size = 64
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device,
        sort_key=lambda x: len(x.text)
    )
    return text, label, train_iterator, valid_iterator, test_iterator


def predict_sentiment(model, sentence, text, min_len=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    nlp = spacy.load('en')
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [text.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


def load_model():
    text, label, _, _, _ = get_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = path.join(path.dirname(__file__), "../saved_models/imdb_cnn.pt")
    input_dim = len(text.vocab)
    embedding_dim = 100
    n_filters = 100
    filter_sizes = [3, 4, 5]
    output_dim = 1
    dropout = 0.5
    pad_idx = text.vocab.stoi[text.pad_token]
    model = CNN(input_dim, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model, text, label


if __name__ == '__main__':
    model, text, label = build_and_train(save=True)
    print(predict_sentiment(model, "This film is terrible", text))
    print(predict_sentiment(model, "This film is great", text))
    #m = load_model()
