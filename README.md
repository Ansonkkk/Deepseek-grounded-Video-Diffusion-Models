# Deepseek-grounded Video Diffusion Models

# Deepseek-grounded Video Diffusion Models (DVD)
The codebase is based on [LLM-grounded Video Diffusion (LVD)](https://github.com/TonyLianLong/LLM-groundedVideoDiffusion) repo, so please also check out the instructions and FAQs there if your issues are not covered in this repo.

## Installation
Install the dependencies:
```
pip install -r requirements.txt
```

## Stage 1: Generating Dynamic Scene Layouts (DSLs) from text
**Note that we have uploaded the layout caches for the benchmark onto this repo so that you can skip this step if you don't need layouts for new prompts (i.e., just want to use LLM-generated layouts for benchmarking stage 2).**

Since we have cached the layout generation (which will be downloaded when you clone the repo), **you need to remove the cache in `cache` directory if you want to re-generate the layout with the same prompts**.

**Our layout generation format:** The LLM takes in a text prompt describing the image and outputs dynamic scene layouts that consist of three elements: **1.** a reasoning statement for analysis, **2.** a captioned box for each object in each frame, and **3.** a background prompt. The template and example prompts are in [prompt.py](prompt.py). You can edit the template, the example prompts, and the parsing function to ask the LLM to generate additional things or even perform chain-of-thought for better generation.

### Automated query from Deepseek API
Again, if you just want to evaluate stage 2 (layout to video stage), you can skip stage 1 as we have uploaded the layout caches onto this repo. **You don't need an Deepseek API key in stage 2.**

If you have an [Deepseek API key](https://www.deepseek.com/), you can put the API key in `utils/api_key.py`. Then you can use  API for batch text-to-layout generation by querying an LLM, with Deepseek as an example:
```shell
python prompt_batch.py --prompt-type demo --model deepseek-chat --auto-query --always-save --template_version v0.1
```
`--prompt-type demo` includes a few prompts for demonstrations. You can change them in [prompt.py](prompt.py). The layout generation will be cached so it does not query the LLM again with the same prompt (lowers the cost).

You can visualize the dynamic scene layouts in the form of bounding boxes in `img_generations/imgs_demo_templatev0.1`. They are saved as `gif` files. For horizontal video generation with zeroscope, the square layout will be scaled according to the video aspect ratio.

### Run our benchmark on text-to-layout generation evaluation
We provide a benchmark that applies both to stage 1 and stage 2. This benchmarks includes a set of prompts with five tasks (numeracy, attribution, visibility, dynamic satial, and sequential) as well as unified benchmarking code for all implemented methods and both stages.

This will generate layouts from the prompts in the benchmark (with `--prompt-type lvd`) and evaluate the results:
```shell
python prompt_batch.py --prompt-type lvd --model deepseek-chat --auto-query --always-save --template_version v0.1
python scripts/eval_stage_one.py --prompt-type lvd --model deepseek-chat --template_version v0.1
```
<details>
<summary>Our reference benchmark results (stage 1, evaluating the generated layouts only)</summary>

| Method      | Numeracy | Attribution | Visibility | Dynamics | Sequential | Overall    |
| --------    | -------- | ----------- | ---------- | -------- | ---------- | ---------- |
| GPT-3.5     | 100      | 100         | 100        | 71       | 16         | 77%        |
| Deepseek-V3 | 100      | 100         | 100        | 84       | 37         | 84%        |
| Deepseek-V3*| 100      | 100         | 100        | **91**   | **41**     | **86%**    |

\* Represents the method we've improved the Prompt.
</details>

## Stage 2: Generating Videos from Dynamic Scene Layouts
Note that since we provide caches for stage 1, you don't need to run stage 1 on your own for cached prompts that we provide (i.e., you don't need an Deepseek API key or to query an LLM).

Similar to LMD+, you can also integrate GLIGEN adapters trained in the [IGLIGEN project](https://github.com/TonyLianLong/igligen) with Modelscope in stage 2. 
For the videos that need to be generated, you only need to modify the corresponding prompt in `prompt.py` and run the following command.
```shell
# Zeroscope (horizontal videos)
python generate.py --model  deepseek-chat --run-model lvd-gligen_zeroscope --prompt-type demo --save-suffix lvd_gligen --template_version v0.1 --seed_offset 0 --repeats 10 --num_frames 24 --gligen_scheduled_sampling_beta 0.4
```
Training-based methods such as GLIGEN typiclly have better spatial control, but sometimes can lead to different interpretations of words w.r.t. the base diffusion model or limited diversity for rare objects. 

## Stage 3: Finetune
The original weights are based on those pre-trained on SA-1B. If you wish to fine-tune them yourself, you can follow the steps below. For details, please refer to [IGLIGEN project](https://github.com/TonyLianLong/igligen)
### Step 1: Data Collection and Preparation
1. **Data Collection**: Search the internet for images of the desired type.
2. **Standardization**: Normalize and rename the images as `image_01`, `image_02`, `image_[number]`, etc.
3. **Packaging**: Compress the images into a single file named `image.tar.gz`.
4. **Annotation**: Manually annotate the images and store the annotations in a CSV file, with each row formatted as `[key caption]`.

Place both the `image.tar.gz` and the annotation CSV file in the `\finetune\igligen\preprocess\` directory.

### Step 2: Environment Setup
Navigate to `\finetune\preprocess\GroundingDINO` and configure the environment. For detailed instructions, refer to the [GroundingDINO project](https://github.com/IDEA-Research/GroundingDINO).

### Step 3: Data Preprocessing
Perform the following preprocessing steps:

1. **Generate Bounding Boxes and Object Descriptions**:
   - Use the configured environment to generate bounding boxes and object descriptions for the images.
```shell
python extract_boxes.py image.tar.gz
```

2. **Convert to Latents**:
   - Transform the processed data into latent representations suitable for model training.
```shell
python encode_latents.py image.tar.gz
```
By following these steps, you will have prepared your dataset for fine-tuning the model. Ensure all prerequisites are met and the environment is correctly configured before proceeding.


## Acknowledgements
This repo is based on [LMD repo](https://github.com/TonyLianLong/LLM-groundedDiffusion), which is based on [diffusers](https://huggingface.co/docs/diffusers/index) and references [GLIGEN](https://github.com/gligen/GLIGEN), [layout-guidance](https://github.com/silent-chen/layout-guidance),[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO). This repo uses the same license as LMD.

