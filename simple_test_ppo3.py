"""
Simple test script to verify ppo_metadrive_best_model.pth can be loaded correctly
"""

import torch
import os
from ppo_metadrive3 import PPOConfig, Actor, Critic, ImageVAE_CNN840Micro_LinearDec, ZValue

def test_model_loading():
    """Test if the model can be loaded correctly"""
    model_path = "ppo_metadrive_best_model.pth"
    
    print(f"Testing model loading: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load checkpoint
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"✅ Checkpoint loaded successfully on {device}")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Initialize networks
        actor = Actor(input_channels=5, z_dim=40).to(device)
        critic = Critic(input_channels=5, z_dim=40).to(device)
        vae_guide = ImageVAE_CNN840Micro_LinearDec(recon_activation=None).to(device)
        z_value = ZValue(z_dim=40).to(device)
        
        # Load state dicts
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        vae_guide.load_state_dict(checkpoint['vae_state_dict'])
        
        # Set to evaluation mode
        actor.eval()
        critic.eval()
        vae_guide.eval()
        z_value.eval()
        
        print("✅ All models loaded successfully!")
        print(f"  - Actor: {sum(p.numel() for p in actor.parameters())} parameters")
        print(f"  - Critic: {sum(p.numel() for p in critic.parameters())} parameters")
        print(f"  - VAE: {sum(p.numel() for p in vae_guide.parameters())} parameters")
        
        # Test forward pass
        print("\nTesting forward pass...")
        dummy_obs = torch.randn(1, 5, 84, 84).to(device)
        
        with torch.no_grad():
            # VAE forward pass
            recon, mu, logvar, z = vae_guide(dummy_obs, return_latent=True)
            z_reshaped = z.view(z.size(0), 20, 2)
            
            # Actor forward pass
            action_dist = actor(dummy_obs, z_reshaped)
            action = action_dist.sample()
            
            # Critic forward pass
            value = critic(dummy_obs, z_reshaped)
            
            print(f"✅ Forward pass successful!")
            print(f"  - VAE output shape: {recon.shape}")
            print(f"  - Latent z shape: {z.shape}")
            print(f"  - Action shape: {action.shape}")
            print(f"  - Value shape: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed! The model is ready for testing.")
    else:
        print("\n❌ Model loading test failed!")
