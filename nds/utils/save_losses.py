import csv
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