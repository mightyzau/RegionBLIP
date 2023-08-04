from torchvision import transforms
from lavis.common.registry import registry 
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("base_point_cloud")
class BasePointCloudProcessor(BaseProcessor):
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __call__(self, item):
        return self.transform(item)