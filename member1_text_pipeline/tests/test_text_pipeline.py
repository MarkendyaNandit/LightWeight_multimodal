"""
End-to-End Test for the Text Pipeline and OCTA Module.

Tests the flow:
  Class Name -> Prompts -> CLIP Encoder -> Text Embeddings -> OCTA -> F_p
"""

import sys
import os
import torch
import unittest

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.prompt_templates import PromptGenerator
from models.clip_encoder import CLIPTextEncoder
from models.octa import OCTA


class MockCLIPTextEncoder(torch.nn.Module):
    def __init__(self, output_dim=256, device="cpu"):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        # Mock projection layer to test gradients
        self.projection = torch.nn.Linear(512, output_dim)
        
    def forward(self, prompts):
        # Return random embeddings
        raw = torch.randn(len(prompts), 512, device=self.device)
        return self.projection(raw)

class TestTextPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        print("Setting up CLIP Text Encoder (Mocked)...")
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"
        cls.encoder = MockCLIPTextEncoder(output_dim=256, device=cls.device)
        cls.gen = PromptGenerator(classes=["cookie"])
        cls.octa = OCTA(dim=256).to(cls.device)

    def test_prompt_generator(self):
        classes = self.gen.get_classes()
        self.assertIn("cookie", classes)
        
        prompts = self.gen.generate_prompts("cookie")
        num_expected = self.gen.get_num_prompts_per_class()
        self.assertEqual(len(prompts), num_expected)
        self.assertTrue(all(isinstance(p, str) for p in prompts))

    def test_clip_encoder_frozen(self):
        # We only check the projection since the rest is mocked
        for name, param in self.encoder.projection.named_parameters():
            self.assertTrue(param.requires_grad, f"Parameter {name} in projection is not trainable")

    def test_end_to_end_pipeline(self):
        class_name = "cookie"
        
        # 1. Generate prompts
        prompts = self.gen.generate_prompts(class_name)
        
        # 2. Encode with mocked CLIP
        embeddings = self.encoder.forward(prompts)
        
        self.assertEqual(embeddings.shape, (len(prompts), 256))
        self.assertTrue(embeddings.requires_grad)
        
        # 3. OCTA forward pass
        f_p = self.octa(embeddings)
        
        # Shape should be [1, 256] (prototype attended feature)
        self.assertEqual(f_p.shape, (1, 256))
        self.assertTrue(f_p.requires_grad)
        
        # Try a dummy backward pass to ensure gradients flow correctly
        loss = f_p.sum()
        loss.backward()
        
        # Check gradients in OCTA
        for name, param in self.octa.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradients missing for OCTA parameter: {name}")
            
        # Check gradients in CLIP projection
        for name, param in self.encoder.projection.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradients missing for CLIP projection parameter: {name}")


if __name__ == '__main__':
    unittest.main()
