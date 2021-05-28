
For visualizations download the Courier Prime Regular font file. See for example: https://fonts.google.com/specimen/Courier+Prime

Run command text8:

``python train.py --batch_size 32 --update_freq 1 --lr 0.0001 --epochs 1000 --eval_every 2 --check_every 20 --diffusion_steps 1000 --transformer_depth 12 --transformer_heads 16 --transformer_local_heads 8 --gamma 0.99``



Run command enwik8:

``python train.py --batch_size 32 --update_freq 1 --lr 0.0001 --epochs 1000 --eval_every 2 --check_every 20 --diffusion_steps 4000 --transformer_depth 12 --data enwik8_blocksparse --transformer_local_size 80 --transformer_heads 16 --transformer_local_heads 8 --gamma 0.99``