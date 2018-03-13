python train.py \
	--save_path saved_checkpoints/model-2 \
	--epochs 35 \
	--loss_margin 1 \
	--dev_every 600 \
	--d_rel_embed 128 \
    	--d_hidden 128 \
	--n_layers 2 \
	--gpu 6 \
	--rnn_type gru 
