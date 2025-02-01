# Leapfusion Hunyuan Image-to-Video
**V2 update!**: [Fal.ai](https://fal.ai/) helped out providing compute to train a better checkpoint! We've just finished training and are happy to release the v2/544p model. This checkpoint should produce significantly better results when compared to the previous. Thanks again to Fal.ai for helping make this possible!

**Show your support!** You can try HunyuanVideo free with some of our custom spice here: [LeapReel](https://leapfusion.ai/reel). Supporting LeapFusion enables us to do more open source releases like this in the future!

Training code can be found [Here](https://github.com/AeroScripts/musubi-tuner-img2video).

# Usage
First, Download the hunyuan weights as explained [here](https://github.com/AeroScripts/musubi-tuner-img2video/tree/main?tab=readme-ov-file#use-the-official-hunyuanvideo-model) and get the image2video lora weights:

[v2 (544p)](https://huggingface.co/leapfusion-image2vid-test/image2vid-960x544/blob/main/img2vid544p.safetensors)
[v1 (320p)](https://huggingface.co/leapfusion-image2vid-test/image2vid-512x320/blob/main/img2vid.safetensors)


Then run the following command to encode an image: (ex. input_image.png)
```
python encode_image.py --vae hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling --video_size 544 960 --image ./input_image.png
```

Then, you can generate a video with something like:
```
python generate.py --fp8 --video_size 544 960 --infer_steps 30 --save_path ./samples/ --output_type both --dit mp_rank_00_model_states.pt --attn_mode sdpa --split_attn --vae hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 --text_encoder1 llava_llama3_fp16.safetensors --text_encoder2 clip_l.safetensors --lora_multiplier 1.0 --lora_weight img2vid544p.safetensors --video_length 85 --prompt "" --seed 123
```
Leaving the prompt blank, the model will infer based on the image alone. If you prompt changes, make sure to describe some baseline details about the image too or you might get bad results.

# Samples (v2 / 544p)

https://github.com/user-attachments/assets/11998d22-18ab-4d2d-b1f5-f38861f867d1

https://github.com/user-attachments/assets/0c570ff4-1e98-4ba8-900b-7405eb1ee44a

https://github.com/user-attachments/assets/64583b7d-8ca6-4752-be29-3ea4a006c00b

https://github.com/user-attachments/assets/a5ee9c3d-8aa7-455c-99c5-d13e6dac88d0

https://github.com/user-attachments/assets/008ec121-2bcb-4056-8880-8e8c0cb2d6db





# Samples (v1 / 320p)
https://github.com/user-attachments/assets/1410ede0-9d88-4c29-b785-bc934525a0da

https://github.com/user-attachments/assets/ef6fcf10-8cdf-42b5-b1fc-d9f8673c894a

https://github.com/user-attachments/assets/30038397-c67c-49f3-9707-db0deb110268

https://github.com/user-attachments/assets/4c53f88e-f44d-45df-81c7-ec3bafc77ec2



## License

Much of the code is based on [musubi-tuner](https://github.com/kohya-ss/musubi-tuner). Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.
Other code is under the Apache License 2.0. Some code is copied and modified from musubi-tuner, k-diffusion and Diffusers.
