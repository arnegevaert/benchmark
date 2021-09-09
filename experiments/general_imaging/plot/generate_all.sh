source ../../../venv/bin/activate

# USAGE: ./generate_all.sh H5_DIR OUT_DIR

# Create output directory if it doesn't exist yet
[ -d "$2" ] || mkdir -p "$2"

# WILCOXON PLOTS
python wilcoxon.py "$1" "$2"/wilcoxon
python wilcoxon.py --all "$1" "$2"/wilcoxon_all

# KRIPPENDORFF ALPHA PLOTS
python krip_alpha_barplot.py "$1" "$2"/krip_bar.png
python krip_alpha_barplot.py --all "$1" "$2"/krip_bar_all.png
# python krip_alpha_variants.py "$1" "$2"/krip_variants

# CORRELATIONS
# python inter_method_correlations.py "$1" "$2"/method_corr
python inter_metric_correlations.py "$1" "$2"/metric_corr
python inter_metric_correlations.py --all "$1" "$2"/metric_corr_all
# python confidence_correlations.py "$1" "$2"/conf_corr
# python confidence_correlations.py --all "$1" "$2"/conf_corr_all

# CLUSTERING
# python cluster.py "$1" "$2"/cluster

# PAIRWISE COMPARISONS
python pairwise_tests.py "$1"/mnist.h5 DeepShap DeepLift "$2"/cles_mnist.png
python pairwise_tests.py "$1"/cifar10.h5 DeepShap DeepLift "$2"/cles_cifar10.png
python pairwise_tests.py "$1"/imagenet.h5 DeepShap DeepLift "$2"/cles_imagenet.png

# MASKING CORRELATIONS
python metric_corr_compare_masking.py "$1" "$2"/metric_corr_masking