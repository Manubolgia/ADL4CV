import torch
import pandas as pd

def equalize_losses(losses_record):
    # Find maximum length among all loss records
    max_len = max(len(record['loss_value']) for record in losses_record)

    # Pad shorter loss records with zeros
    for record in losses_record:
        record_len = len(record['loss_value'])
        if record_len < max_len:
            record['loss_value'].extend([0.0]*(max_len - record_len))
    return losses_record


def save_losses(losses_record, plots_save_path):
    # Convert to DataFrame
    losses_record = equalize_losses(losses_record)
    df = pd.DataFrame(losses_record)

    # Save to csv
    df.to_csv(plots_save_path / 'losses.csv', index=False)

    print('Losses saved correctly')


def print_cuda_memory(iteration, frequency=100):
    if iteration % frequency == 0:
        current = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        remaining = total - current
        print(f"\nAfter iteration {iteration}:")
        print(f"Current GPU memory usage: {current / 1024**2} MB")
        print(f"Peak GPU memory usage: {peak / 1024**2} MB")
        print(f"Total GPU memory: {total / 1024**2} MB")
        print(f"Remaining GPU memory: {remaining / 1024**2} MB")

