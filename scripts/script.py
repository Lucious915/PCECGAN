import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=str, default="8097")
parser.add_argument("--train", action='store_true')
parser.add_argument("--predict", action='store_true')
opt = parser.parse_args()

if opt.train:
	os.system("python train.py \
		--dataroot ../final_dataset \
		--no_dropout \
		--name enlightening \
		--model ORCGAN \
		--dataset_mode unaligned \
		--which_model_netG orcgan_unet_resize \
        --which_model_netD no_norm_4 \
        --patchD \
        --patch_vgg \
        --patchD_3 5 \
        --n_layers_D 5 \
        --n_layers_patchD 4 \
		--fineSize 150 \
        --patchSize 32 \
		--skip 1 \
		--batchSize 10 \
        --self_attention \
		--use_norm 1 \
		--use_wgan 0 \
        --use_ragan \
        --hybrid_loss \
        --times_residual \
		--instance_norm 0 \
		--vgg 1 \
        --vgg_choose relu5_1 \
		--gpu_ids 0 \
		--linear_add \
		--power_constraint\
		--display_port=" + opt.port)
		#--continue_train\
		#--which_epoch 25\
		
	
	
		

elif opt.predict:
	for i in range(1):
	        os.system("python predict.py \
	        	--dataroot ../final_dataset \
	        	--name enlightening \
	        	--model ORCGAN \
	        	--which_direction AtoB \
	        	--no_dropout \
	        	--dataset_mode unaligned \
	        	--which_model_netG orcgan_unet_test_resize \
	        	--skip 1 \
	        	--use_norm 1 \
	        	--use_wgan 0 \
                --self_attention \
                --times_residual \
	        	--instance_norm 0 \
	        	--resize_or_crop='no'\
	        	--output_std 5\
	        	--output_mean 0.9\
	        	--linear_add \
	        	--which_epoch 200" ) # + str(200 - i*5)15_enlightening_ablation_power_constraint
	        

	        	# --power_constraint\