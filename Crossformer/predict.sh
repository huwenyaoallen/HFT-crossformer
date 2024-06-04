
python main_crossformer.py \
--root_path ./30min_datasets \
--data_split 0.7,0.1,0.2 \
--in_len 96 \
--seg_len 4 \
--data_dim 13 \
--scale_factor 50 \
--train_epochs 20 \
--learning_rate 5e-5 \
--gpu 0 \

python main_mlp.py \
--root_path ./30min_datasets \
--data_split 0.7,0.1,0.2 \
--in_len 96 \
--data_dim 13 \
--scale_factor 50 \
--train_epochs 20 \
--learning_rate 5e-5 \
--gpu 0 \

python main_transformer.py \
--root_path ./30min_datasets \
--data_split 0.7,0.1,0.2 \
--in_len 96 \
--seg_len 4 \
--data_dim 13 \
--scale_factor 50 \
--train_epochs 20 \
--learning_rate 5e-5 \
--gpu 0 \