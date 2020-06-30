import torch
from util.datasets import Aptos
import numpy as np
import time
from sklearn.metrics import cohen_kappa_score, confusion_matrix, plot_confusion_matrix
from util.models import AptosDensenet


if __name__ == '__main__':
    aptos = Aptos(batch_size=8, img_size=320, data_location="../../../data/Aptos")
    model = AptosDensenet(output_logits=False, params_loc="../../../data/models/aptos_densenet_121.pt")
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    pred = []
    true_classes = []
    start_time = time.time()
    with torch.no_grad():
        for data, targets in aptos.get_test_data():
            data, targets = torch.tensor(data, dtype=torch.float).cuda(non_blocking=True), \
                            torch.tensor(targets, dtype=torch.long).cuda(non_blocking=True)
            data, targets = data.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            out = model(data)
            pred.extend(out.cpu().numpy())
            true_classes.extend(targets.cpu().numpy())
    stop_time = time.time()
    print(f"Elapsed time: {stop_time - start_time:0.4f}")
    pred = np.array(pred)
    true_classes = np.array(true_classes)
    rounded_predictions = np.argmax(pred, axis=1)

    print('quadratic weighted Kappa score: ', cohen_kappa_score(true_classes, rounded_predictions, weights='quadratic'))
    cnf_matrix = confusion_matrix(true_classes, rounded_predictions)
    print(cnf_matrix)
