import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = "resnet101deeplab_ver_3.pth"
device = torch.device('cpu')
# model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model = models.segmentation.deeplabv3_resnet101(weights=None, progress=True, num_classes=21)
# model = models.segmentation.deeplabv3_resnet101(pretrained)
# model.load_state_dict(torch.load(FILE, map_location=device))
model.eval().to(device)

# example = torch.rand(10,3,300,300)
script_module = torch.jit.script(model)
optimized_script_module = optimize_for_mobile(script_module)
optimized_script_module.save("assets/model.pt")
optimized_script_module._save_for_lite_interpreter("assets/model.ptl")

# example = torch.rand(10,3,300,300)
# traced_script_module = torch.jit.trace(model, example,strict=False)
# optimized_traced_model = optimize_for_mobile(traced_script_module)
# optimized_traced_model.save("assets/model.pt")
# optimized_traced_model._save_for_lite_interpreter("assets/model.ptl")