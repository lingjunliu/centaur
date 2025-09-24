## Coverage

# Pytorch
python -m utils.misc plots/data/titanfuzz_180s.csv plots/data/slate_html_180s.csv plots/data/vs_Titanfuzz.csv
python -m utils.misc plots/data/acetest_180s.csv plots/data/slate_html_180s.csv plots/data/vs_ACETest.csv
python -m utils.misc plots/data/pathfinder_180s.csv plots/data/slate_html_180s.csv plots/data/vs_Pathfinder.csv

# Tensoflow
python -m utils.misc plots/data/titanfuzz_180s_tf.csv plots/data/slate_html_180s_tf.csv plots/data/vs_Titanfuzz_tf.csv
python -m utils.misc plots/data/acetest_180s_tf.csv plots/data/slate_html_180s_tf.csv plots/data/vs_ACETest_tf.csv
python -m utils.misc plots/data/pathfinder_180s_tf.csv plots/data/slate_html_180s_tf.csv plots/data/vs_Pathfinder_tf.csv

cd plots

# ACETest Pytorch
Rscript invariants.r ACETest cov data/vs_ACETest.csv > data/ACETest_stats.txt
# Pathfinder Pytorch
Rscript invariants.r Pathfinder cov data/vs_Pathfinder.csv > data/Pathfinder_stats.txt
# Titanfuzz Pytorch
Rscript invariants.r Titanfuzz cov data/vs_Titanfuzz.csv > data/Titanfuzz_stats.txt

# ACETest Tensorflow
Rscript invariants.r ACETest cov data/vs_ACETest_tf.csv tf > data/ACETest_stats_tf.txt
# Pathfinder Tensorflow
Rscript invariants.r Pathfinder cov data/vs_Pathfinder_tf.csv tf > data/Pathfinder_stats_tf.txt
# Titanfuzz Tensorflow
Rscript invariants.r Titanfuzz cov data/vs_Titanfuzz_tf.csv tf > data/Titanfuzz_stats_tf.txt

nAPIsTorch=$(cat data/slate_html_180s.csv | wc -l)
nAPIsTF=$(cat data/slate_html_180s_tf.csv | wc -l)

nAPIsSLTorch=$((nAPIsTorch - 1))
nAPIsSLTF=$((nAPIsTF - 1))

echo "% Pytorch" > data/Coverage_stats.txt && echo "\newcommand{\nAPIsSL}{${nAPIsSLTorch}}" >> data/Coverage_stats.txt  && cat data/Titanfuzz_stats.txt data/ACETest_stats.txt data/Pathfinder_stats.txt >> data/Coverage_stats.txt && echo "% Tensorflow" >> data/Coverage_stats.txt && echo "\newcommand{\nAPIsSLtf}{${nAPIsSLTF}}" >> data/Coverage_stats.txt && cat data/Titanfuzz_stats_tf.txt data/ACETest_stats_tf.txt data/Pathfinder_stats_tf.txt >> data/Coverage_stats.txt

cd ..

## Validity
# Pytorch
python -m utils.misc plots/data/titanfuzz_val_torch.csv plots/data/slate_val_torch.csv plots/data/val_vs_Titanfuzz.csv
python -m utils.misc plots/data/acetest_val_torch.csv plots/data/slate_val_torch.csv plots/data/val_vs_ACETest.csv
python -m utils.misc plots/data/pathfinder_val_torch.csv plots/data/slate_val_torch.csv plots/data/val_vs_Pathfinder.csv

# Tensoflow
python -m utils.misc plots/data/titanfuzz_val_tf.csv plots/data/slate_val_tf.csv plots/data/val_vs_Titanfuzz_tf.csv
python -m utils.misc plots/data/acetest_val_tf.csv plots/data/slate_val_tf.csv plots/data/val_vs_ACETest_tf.csv
python -m utils.misc plots/data/pathfinder_val_tf.csv plots/data/slate_val_tf.csv plots/data/val_vs_Pathfinder_tf.csv

cd plots

# ACETest Pytorch
Rscript invariants.r ACETest val data/val_vs_ACETest.csv > data/ACETest_stats_val.txt
# Pathfinder Pytorch
Rscript invariants.r Pathfinder val data/val_vs_Pathfinder.csv > data/Pathfinder_stats_val.txt
# Titanfuzz Pytorch
Rscript invariants.r Titanfuzz val data/val_vs_Titanfuzz.csv > data/Titanfuzz_stats_val.txt

# ACETest Tensorflow
Rscript invariants.r ACETest val data/val_vs_ACETest_tf.csv tf > data/ACETest_stats_tf_val.txt
# Pathfinder Tensorflow
Rscript invariants.r Pathfinder val data/val_vs_Pathfinder_tf.csv tf > data/Pathfinder_stats_tf_val.txt
# Titanfuzz Tensorflow
Rscript invariants.r Titanfuzz val data/val_vs_Titanfuzz_tf.csv tf > data/Titanfuzz_stats_tf_val.txt

nAPIsTorch=$(cat data/slate_val_torch.csv | wc -l)
nAPIsTF=$(cat data/slate_val_tf.csv | wc -l)

nAPIsSLTorch=$((nAPIsTorch - 1))
nAPIsSLTF=$((nAPIsTF - 1))

echo "% Pytorch" > data/Validity_stats.txt && echo "\newcommand{\nAPIsSL}{${nAPIsSLTorch}}" >> data/Validity_stats.txt  && cat data/Titanfuzz_stats_val.txt data/ACETest_stats_val.txt data/Pathfinder_stats_val.txt >> data/Validity_stats.txt && echo "% Tensorflow" >> data/Validity_stats.txt && echo "\newcommand{\nAPIsSLtf}{${nAPIsSLTF}}" >> data/Validity_stats.txt && cat data/Titanfuzz_stats_tf_val.txt data/ACETest_stats_tf_val.txt data/Pathfinder_stats_tf_val.txt >> data/Validity_stats.txt

# Validity Ratio Table
python validity_table.py