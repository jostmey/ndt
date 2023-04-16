mkdir -p ../results

python3 run_NDT.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/ndt_bank-marketing.p > ../results/ndt_bank-marketing.txt
python3 run_NDT.py --device gpu --dataset ../datasets/census-income/ --output ../results/ndt_census-income.p > ../results/ndt_census-income.txt
python3 run_NDT.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/ndt_mnist-odd.p > ../results/ndt_mnist-odd.txt

python3 run_NST.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/nst_bank-marketing.p > ../results/nst_bank-marketing.txt
python3 run_NST.py --device gpu --dataset ../datasets/census-income/ --output ../results/nst_census-income.p > ../results/nst_census-income.txt
python3 run_NST.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/nst_mnist-odd.p > ../results/nst_mnist-odd.txt

python3 run_NGT.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/ngt_bank-marketing.p > ../results/ngt_bank-marketing.txt
python3 run_NGT.py --device gpu --dataset ../datasets/census-income/ --output ../results/ngt_census-income.p > ../results/ngt_census-income.txt
python3 run_NGT.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/ngt_mnist-odd.p > ../results/ngt_mnist-odd.txt

python3 run_BLR.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/blr_bank-marketing.p > ../results/blr_bank-marketing.txt
python3 run_BLR.py --device gpu --dataset ../datasets/census-income/ --output ../results/blr_census-income.p > ../results/blr_census-income.txt
python3 run_BLR.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/blr_mnist-odd.p > ../results/blr_mnist-odd.txt

python3 run_FCNN.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/fcnn_bank-marketing.p > ../results/fcnn_bank-marketing.txt
python3 run_FCNN.py --device gpu --dataset ../datasets/census-income/ --output ../results/fcnn_census-income.p > ../results/fcnn_census-income.txt
python3 run_FCNN.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/fcnn_mnist-odd.p > ../results/fcnn_mnist-odd.txt

python3 run_FFT.py --device gpu --dataset ../datasets/bank-marketing/ --output ../results/fft_bank-marketing.p > ../results/fft_bank-marketing.txt
python3 run_FFT.py --device gpu --dataset ../datasets/census-income/ --output ../results/fft_census-income.p > ../results/fft_census-income.txt
python3 run_FFT.py --device gpu --dataset ../datasets/mnist-odd/ --output ../results/fft_mnist-odd.p > ../results/fft_mnist-odd.txt
