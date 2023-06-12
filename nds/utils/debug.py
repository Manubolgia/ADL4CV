import csv
import torch

def save_losses(losses_record, plots_save_path):

    # convert tensor to numpy array
    losses_record = losses_record.detach().cpu().numpy()

    # open the file in write mode
    with open(plots_save_path / 'losses.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        # write rows
        for row in losses_record:
            writer.writerow(row)

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

