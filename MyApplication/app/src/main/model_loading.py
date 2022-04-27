import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

FILE = "C:/Users/titig/AprilPush/CapStoneProject/MyApplication/app/src/main/resnet50_ver_1_1.pth"

resnet50fcn = models.segmentation.fcn_resnet50(pretrained=False, progress=True, num_classes=21)
resnet50fcn.load_state_dict(torch.load(FILE, map_location=torch.device('cpu')))
resnet50fcn.eval()

example = torch.rand(10,3,300,300)
traced_script_module = torch.jit.trace(resnet50fcn, example,strict=False)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("C:/Users/titig/AprilPush/CapStoneProject/MyApplication/app/src/main/assets/model3.pth")