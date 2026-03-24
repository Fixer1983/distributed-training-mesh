
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    model = SimpleModel().to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    
    # Simulated training loop
    for i in range(10):
        optimizer.zero_grad()
        outputs = ddp_model(torch.randn(20, 10).to(rank))
        labels = torch.randn(20, 1).to(rank)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Step {i}, Loss: {loss.item()}")
            
    cleanup()

if __name__ == "__main__":
    # In practice, this would be launched via torchrun
    print("Distributed training mesh initialized.")
