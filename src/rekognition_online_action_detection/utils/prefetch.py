import torch

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.nextitem = next(self.loader)
            self.fusion_visual_inputs = self.nextitem[0]
            self.fusion_motion_inputs = self.nextitem[1]
            self.memory_key_padding_mask = self.nextitem[2]
            self.target_current = self.nextitem[3]
            self.target_long = self.nextitem[4]
        except StopIteration:
            self.fusion_visual_inputs = None
            self.fusion_motion_inputs = None
            self.memory_key_padding_mask = None
            self.target_current = None
            self.target_long = None
            return
        with torch.cuda.stream(self.stream):
            self.fusion_visual_inputs = self.fusion_visual_inputs.cuda(non_blocking=True)
            self.fusion_motion_inputs = self.fusion_motion_inputs.cuda(non_blocking=True)
            self.memory_key_padding_mask = self.memory_key_padding_mask.cuda(non_blocking=True)
            self.target_current = self.target_current.cuda(non_blocking=True)
            self.target_long = self.target_long.cuda(non_blocking=True)
 
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        fusion_visual_inputs = self.fusion_visual_inputs
        fusion_motion_inputs = self.fusion_motion_inputs
        memory_key_padding_mask = self.memory_key_padding_mask
        target_current = self.target_current
        target_long = self.target_long
        self.preload()
        return fusion_visual_inputs, fusion_motion_inputs, memory_key_padding_mask, target_current, target_long