

CUDA_VISIBLE_DEVICES=0 

#--use-flash-attention
#--use-sage-attention
python main.py \
	--enable-manager \
	--enable-cors-header \
	--listen 0.0.0.0 \
	--port 9901 \
	--cuda-device $CUDA_VISIBLE_DEVICES \
	--force-fp16 \
	--fp8_e4m3fn-unet \
	--bf16-vae \
	--fp8_e4m3fn-text-enc \
	--supports-fp8-compute \
	--cache-lru 100 \
	--normalvram \
	--fast \
	--use-flash-attention \
	--mmap-torch-files 
