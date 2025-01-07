# Temporal LM: Benchmarking LMs for TKG Link Prediction <br>

## Dataset: <br>

Original Dataset: <br>
https://drive.google.com/drive/folders/1lfxUw8sRuL5qDYlX42Z-AFsgbgIVQhyJ

Dataset after initial splitting: <br>
https://drive.google.com/drive/folders/1uP66nWurssE9Wn-tZdbl4OTECC6o2_ro?usp=sharing


Command to run the experiments:

python run_experiments.py --do_train --do_test --data_dir ./data/YAGO11k --dataset_name yago --epochs 50 --sampling 0.01 --results_save_dir ./Experiments_Results --tensorboard True --batch 1024 --n_temporal_neg 1 --lr 0.001 --min_time 0 --max_time 100 --margin 1.0 --save_to_dir Trained_Models --use_descriptions

python link_prediction.py --data_dir "./data/inductive/all-triples/YAGO11k" --do_train --epochs 5 --batch_size 1024 --do_test --lr 0.001 --save_model --save_to "ind_yago11k_tp_model.pth" --use_descriptions --min_time -453 --max_time 2844 --n_temporal_neg 1 --embedding_model "all_mpnet_base_v2" --sampling 0.05 --results_save_dir "./yago_results/all_mpnet_base_v2" --tensorboard_log_dir "./yago_mpnet_logs"