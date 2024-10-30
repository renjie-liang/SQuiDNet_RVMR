import os
import torch
import torch.nn as nn
import numpy as np
import deepspeed
import torch.distributed as dist

print(f'TORCH_CUDA_ARCH_LIST: {os.environ.get("TORCH_CUDA_ARCH_LIST")}')


# Set up a basic linear regression model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.criterion = nn.MSELoss()
    def forward(self, x, y):
        x = self.linear(x)
        loss = self.criterion(x, y)
        return loss

# Create a synthetic dataset
def generate_data(num_samples, input_dim):
    X = np.random.rand(num_samples, input_dim)
    y = np.random.rand(num_samples, 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Configuration for DeepSpeed
# deepspeed_config =  {
#         "train_batch_size": 8,
#             "fp16": {
#                 "enabled": True,
#                 "loss_scale": 0,
#                 "loss_scale_window": 1000,
#                 "hysteresis": 2,
#                 "min_loss_scale": 1
#             },

#         "zero_optimization": {
#             "stage": 2,
#             "allgather_partitions": True,
#             "allgather_bucket_size": 2e8,
#             "overlap_comm": True,
#             "reduce_scatter": True,
#             "reduce_bucket_size": 2e8,
#             "contiguous_gradients": True,
#             "cpu_offload": False
#         },

#         "optimizer": {
#             "type": "AdamW",
#             "params": {
#             "lr": 3e-5,
#             "betas": [
#                 0.8,
#                 0.999
#             ],
#             "eps": 1e-8,
#             "weight_decay": 3e-7
#             }
#         },

#         "scheduler": {
#             "type": "WarmupLR",
#             "params": {
#             "warmup_min_lr": 0,
#             "warmup_max_lr": 3e-5,
#             "warmup_num_steps": 500
#             }
#         },

#             "steps_per_print": 2000,
#             "wall_clock_breakdown": False
#         }

deepspeed_config = {
                "train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.00015
                }
                },
                "fp16": {
                "enabled": True
                },
                "session_params": {
                    "zero_optimization": {
                        "stage": 1,
                        }                
            }
        }

def main():
    input_dim = 10
    output_dim = 1
    num_samples = 100
    batch_size = 8
    num_epochs = 5

    device = torch.device('cuda')
    model = SimpleModel(input_dim, output_dim)
    # Generate data
    X, y = generate_data(num_samples, input_dim)
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=deepspeed_config)

    # model.train()
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # outputs = model_engine(inputs)
            # loss = criterion(outputs, targets)
            # model_engine.backward(loss)
            # model_engine.step()

            loss = model_engine(inputs, targets)
            # loss = criterion(outputs, targets)
            model_engine.backward(loss)
            model_engine.step()

            # Forward pass
            # outputs = model(inputs)
            # loss = criterion(outputs, targets)

            # # Backward and optimize
            # model.zero_grad()
            # loss.backward()
            # optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=5,7 deepspeed --num_gpus=1 demo_deepspeed.py
# CUDA_VISIBLE_DEVICES=5,7 deepspeed --hostfile config/hostfile.txt --num_gpus=2 demo_deepspeed.py
# deepspeed --hostfile hostfile.txt --num_gpus=1 demo_deepspeed.py
# deepspeed  --num_gpus=1 demo_deepspeed.py