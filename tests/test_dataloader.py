import torch
from torch.utils.data import TensorDataset, DataLoader
from syckpt.dataloader import StatefulRandomSampler

def test_sampler_exact_resumption_slice():
    # 1. Create dummy data
    data = torch.arange(0, 100)
    dataset = TensorDataset(data)
    
    # 2. Initialize our specialized sampler
    sampler = StatefulRandomSampler(dataset, batch_size=10, base_seed=42)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)
    
    # Run the first three batches
    iterator = iter(loader)
    b1 = next(iterator)
    sampler.batch_idx += 1
    
    b2 = next(iterator)
    sampler.batch_idx += 1
    
    b3 = next(iterator)
    sampler.batch_idx += 1
    
    # Save the exact internal state
    crash_state = sampler.state_dict()
    
    # Assume a crash happens here. What is the ground-truth target for batch 4?
    ground_truth_b4 = next(iterator)
    
    # 3. Simulate process restart
    resumed_sampler = StatefulRandomSampler(dataset, batch_size=10, base_seed=42)
    resumed_sampler.load_state_dict(crash_state)
    
    resumed_loader = DataLoader(dataset, batch_size=10, sampler=resumed_sampler)
    resumed_iterator = iter(resumed_loader)
    
    # 4. Assert O(1) mathematical alignment without any iteration skips!
    resumed_b4 = next(resumed_iterator)
    
    assert torch.equal(ground_truth_b4[0], resumed_b4[0]), "Fast-forward slice failed determinism."
        
if __name__ == "__main__":
    test_sampler_exact_resumption_slice()
    print("test_sampler_exact_resumption_slice PASSED.")
