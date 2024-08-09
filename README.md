# LayoutEditing

# Installation
Depending on your CUDA version you need to install a different environment.
The one using CUDA 12.X is the only one supporting FLASH ATTENTION 2.
## CUDA 11.X
    conda env create -f environment.yml
    conda activate LayoutEditing

## CUDA 12.X 
    conda env create -f environment3.yml
    conda activate LayoutEditing2

## Usage
Arguments for the main.py file:
```
    --no-guide: Unguided generation
    --guide : Guides the generation with attention control
    --method " new | pww | zero_shot " : Choose guidance method
    --diffusion_type " SD | LCM " : Choose diffusion model
    --res " 16 | 32 | 64 " : Attention resolution size
    --out_dir <output_path> : Output folder
    --cuda <gpu_id> : What gpu to use
    --eta <eta> : Sets eta value from the paper
    --prompt <image_prompt> : Image prompt
    --seed <seed> : sets seed
    --mask_index <number> : Sets which word of the prompt is associated with the given attention
    --mask <path> : Sets the attention image for guidance
    --from_file <*.json> : Loads a json file with dataset information for generation
```
### Singe Image LayoutEditing with SD 1.5 

```
python3 main.py --guide --method new --diffusion_type SD --res 32 --out_dir out/  --cuda 0 --eta 0.8 --prompt "A man with a red shirt and green hair near a woman with a blue dress and blonde hair" --seed 32 --mask_index 8 --mask "attention/green.png"
```

### Guidance on COCO dataset
First generate attention masks based on the datasets' annotations
```
 python3 coco_segmentation_eval.py --task generate_mask --output_path attentions/
```
This will generate a file called eval.json containing data about the dataset.
Then start guided generation on the whole dataset with:
```
python3 main.py --guide --method new --diffusion_type SD --from_file eval.json --res 32 --out_dir results/  --cuda 0 --eta 0.01 
```

### Benchmarking the results
To compute CLIP scores:
```
python3 coco_segmentation_eval.py --task clip --output_path results/
```
To compute iou, first download a SAM model, in the paper we use the VIT-H model [SAM](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and put it into the ```checkpoints/``` folder then run:
```
python3 coco_segmentation_eval.py --task iou --cuda 0 --output_path results/
```
