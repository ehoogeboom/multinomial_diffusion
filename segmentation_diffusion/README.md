Unfortunately due to the license of Cityscapes we cannot include the preprocessed dataset here. To run the experiment first download the data from https://www.cityscapes-dataset.com and run ``data_to_npy.py``. 


Then, to run the model:

``python train.py --eval_every 1 --check_every 25 --epochs 500 --log_home /var/scratch/ehoogebo --diffusion_steps 4000 --dataset cityscapes_coarse --batch_size 32 --dp_rate 0.1 --augmentation shift --lr 0.0001 --warmup 5 --batch_size 64``


Note that what is now line 194 in layer.py containing the UNet definition has changed. An unnatural permute is now operating on the correct axes, thanks to denkorzh for pointing this out. This also means that the Multinomial Diffusion model performs on par with the flow-based approaches with 0.304 bpd on the cityscapes experiment.