source ../../../venv/bin/activate

python wilcoxon.py ../../../out out/wilcoxon
python krip_alpha_barplot.py ../../../out out/krip_bar.png --datasets mnist cifar10 imagenet
python krip_alpha_lineplot.py ../../../out out/krip_line
python inter_method_correlations.py ../../../out out/method_corr
python inter_metric_correlations.py ../../../out out/metric_corr
python cluster.py ../../../out out/cluster
