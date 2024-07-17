# run: export LD_LIBRARY_PATH=/workspace/InstructBLIP-TPU/LLM-TPU/support/lib_pcie
# ln -s /workspace/libsophon-0.5.1/ /opt/sophon/libsophon-current
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("../../docs/_static/Confusing-Pictures.jpg").convert("RGB")

import sys
sys.path.append("../../")

from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
breakpoint()
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct_tpu", model_type="vicuna7b", is_eval=True, device=device)

# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# breakpoint()
# o1 = model.generate({"image": image, "prompt": "What is unusual about this image?"})
# breakpoint()
# o2 = model.generate({"image": image, "prompt": "Write a detailed description."})
breakpoint()
o3 = model.generate({"image": image, "prompt":"Describe the image in details."}, use_nucleus_sampling=True, top_p=0.9, temperature=1)
breakpoint()
