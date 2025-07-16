# ACETest Pytorch
Rscript invariants.r ACETest cov data/vs_ACETest.csv > data/ACETest_stats.txt
# Pathfinder Pytorch
Rscript invariants.r Pathfinder cov data/vs_Pathfinder.csv > data/Pathfinder_stats.txt
# Titanfuzz Pytorch
Rscript invariants.r Titanfuzz cov data/vs_titanfuzz.csv > data/Titanfuzz_stats.txt

# ACETest Tensorflow
Rscript invariants.r ACETest cov data/vs_ACETest_tf.csv tf > data/ACETest_stats_tf.txt
# Pathfinder Tensorflow
Rscript invariants.r Pathfinder cov data/vs_Pathfinder_tf.csv tf > data/Pathfinder_stats_tf.txt
# Titanfuzz Tensorflow
Rscript invariants.r Titanfuzz cov data/vs_titanfuzz_tf.csv tf > data/Titanfuzz_stats_tf.txt