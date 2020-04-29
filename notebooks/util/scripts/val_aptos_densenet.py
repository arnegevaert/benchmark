import torch
from datasets import Aptos
import numpy as np
import time
from sklearn.metrics import cohen_kappa_score, confusion_matrix, plot_confusion_matrix
from models.aptos_densenet import AptosDensenet




if __name__ == '__main__':

    Aptos_data = Aptos(batch_size=8, img_size=320)
    # model = torch.load('./best_model_checkpoint.pt')
    model = AptosDensenet(densenet="densenet121")
    model.net.load_state_dict(torch.load("./best_model_checkpoint.pt"))
    dl_train = Aptos_data.get_train_data()
    dl_val = Aptos_data.get_test_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pred = []
    true_classes = []
    start_time = time.time()
    with torch.no_grad():
        for data, targets in dl_val:
            data, targets = data.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            out = model.predict(data)
            pred.extend(out.cpu().numpy())
            true_classes.extend(targets.cpu().numpy())
    stop_time = time.time()
    print(f"Elapsed time: {stop_time - start_time:0.4f}")
    pred = np.array(pred)
    true_classes = np.array(true_classes)
    rounded_predictions = np.argmax(pred, axis=1)

    # %%

    print('quadratic weighted Kappa score: ', cohen_kappa_score(true_classes, rounded_predictions, weights='quadratic'))
    cnf_matrix = confusion_matrix(true_classes, rounded_predictions)
    print(cnf_matrix)
