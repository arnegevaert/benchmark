source ../../../venv/bin/activate

# python wilcoxon.py ../../../out out/wilcoxon
python wilcoxon.py --all ../../../out out/wilcoxon_all

# python krip_alpha_barplot.py ../../../out out/krip_bar.png
python krip_alpha_barplot.py --all ../../../out out/krip_bar_all.png

# python krip_alpha_lineplot.py ../../../out out/krip_line
# python krip_alpha_variants.py ../../../out out/krip_variants

# python inter_method_correlations.py ../../../out out/method_corr

# python inter_metric_correlations.py ../../../out out/metric_corr
python inter_metric_correlations.py --all ../../../out out/metric_corr_all

# python cluster.py ../../../out out/cluster

python confidence_correlations.py --all ../../../out out/conf_corr