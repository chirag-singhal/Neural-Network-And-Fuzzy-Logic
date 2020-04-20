import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T


resize = T.Compose([
	T.ToPILImage(),
	T.Resize(100, interpolation=Image.CUBIC),
	T.ToTensor()])

def get_screen(env, device):
	# Returned screen requested by gym is 400x600x3, but is sometimes larger
	# such as 800x1200x3. Transpose it into torch order (CHW)
	screen = env.render(mode='rgb_array').transpose((2, 0, 1))

	# Following code crops unwanted vertical portion of image for 'VideoPinball-v0'	
	_, screen_height, screen_width = screen.shape
	screen = screen[:, int(screen_height*0.19):int(screen_height*(1 - 0.12))]

	# Convert to float, rescale, convert to torch tensor
	# (this doesn't require a copy)
	screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
	screen = torch.from_numpy(screen)
	# Resize, and add a batch dimension (BCHW)
	return resize(screen).unsqueeze(0).to(device)
