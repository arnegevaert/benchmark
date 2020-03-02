from os import path
from models.scripts.train_imdb_cnn import load_model, predict_sentiment
import torch
from captum.attr import TokenReferenceBase, configure_interpretable_embedding_layer, IntegratedGradients

# source: https://captum.ai/tutorials/IMDB_TorchText_Interpret


if __name__ == '__main__':
    model_path = path.join(path.dirname(__file__), "../models/saved_models/imdb_cnn.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = path.join(path.dirname(__file__), "../../data/imdb_preproc")

    model, text, label = load_model()
    pos_result = predict_sentiment(model, "This film is great", text)
    neg_result = predict_sentiment(model, "This film is terrible", text)
    print(f"Positive review: {pos_result}")
    print(f"Negative review: {neg_result}")

    pad_idx = text.vocab.stoi[text.pad_token]
    token_reference = TokenReferenceBase(reference_token_idx=pad_idx)
    interpretable_emb = configure_interpretable_embedding_layer(model, "embedding")

    # See also https://captum.ai/tutorials/Multimodal_VQA_Interpret
