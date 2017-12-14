python train.py --rel_vocab_file ../vocab/vocab.rel.sep.pt \
	--save_path saved_checkpoints/model-6 \
	--train_file data/train.relation_ranking.separated.pt \
	--valid_file data/valid.relation_ranking.separated.pt \
	--loss_margin 1 \
	--dev_every 600 \
	--rnn_type gru
