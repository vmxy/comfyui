
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0 
export CUDNN_V8_API_ENABLED=1

#--use-flash-attention
#--use-sage-attention
python main.py \
	--cuda-device $CUDA_VISIBLE_DEVICES \
	--port 9901 \
	--listen 0.0.0.0 \
	--enable-manager \
	--enable-cors-header \
	--force-fp16 \
	--fp8_e4m3fn-unet \
	--bf16-vae \
	--fp8_e4m3fn-text-enc \
	--supports-fp8-compute \
	--cache-lru 32 \
	--normalvram \
	--fast fp16_accumulation fp8_matrix_mult cublas_ops autotune \
	--use-sage-attention \
	--mmap-torch-files 
