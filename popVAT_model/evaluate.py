import numpy as np
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

def calculate_district_mse(predicted_image, district_masks, ins_population):
    true_values_all_districts = []
    predicted_values_all_districts = []

    for district in range(len(district_masks)):
        district_mask = district_masks[district] > 0

        # Align the shape of district_mask to predicted_image if necessary
        if district_mask.shape != predicted_image.shape:
            if district_mask.shape[0] > predicted_image.shape[0]:
                district_mask = district_mask[:predicted_image.shape[0], :]
            if district_mask.shape[1] > predicted_image.shape[1]:
                district_mask = district_mask[:, :predicted_image.shape[1]]

        predicted_population_district = np.sum(predicted_image * district_mask)
        true_population_district = ins_population[district]
        true_values_all_districts.append(true_population_district)
        predicted_values_all_districts.append(predicted_population_district)

    mse_district = mean_squared_error(true_values_all_districts, predicted_values_all_districts)
    print(f"District-level MSE: {mse_district}")
    return mse_district


def calculate_pixel_r2(target_data, predicted_image):
    y_true = np.expm1(target_data).ravel()
    y_pred = predicted_image.ravel()
    r2_pixel = r2_score(y_true, y_pred)
    print(f"Pixel-level R² Score: {r2_pixel}")
    return r2_pixel

def calculate_district_r2(predicted_image, district_masks, ins_population):
    true_values_all_districts = []
    predicted_values_all_districts = []

    for district in range(len(district_masks)):
        district_mask = district_masks[district] > 0

        # Align district_mask to predicted_image
        min_rows = min(predicted_image.shape[0], district_mask.shape[0])
        min_cols = min(predicted_image.shape[1], district_mask.shape[1])
        district_mask = district_mask[:min_rows, :min_cols]
        image_crop = predicted_image[:min_rows, :min_cols]


        total = np.sum(predicted_image)
        predicted_population_district = np.sum(image_crop  * district_mask)
        true_population_district = ins_population[district]
        true_values_all_districts.append(true_population_district)
        predicted_values_all_districts.append(predicted_population_district)
        print(f"total: {total}")
        print(f"district: {district}")
        print(f"predicted: {predicted_population_district}")
        print(f"true: {true_population_district}")

    r2_district = r2_score(true_values_all_districts, predicted_values_all_districts)
    print(f"District-level R² Score: {r2_district}")
    return r2_district


def save_plots(history, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'loss.png'))

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'mae.png'))

    plt.show()

def calculate_district_mae(predicted_image, district_masks, ins_population):
    true_values_all_districts = []
    predicted_values_all_districts = []

    for district in range(len(district_masks)):
        district_mask = district_masks[district] > 0

        # Align the shape of district_mask to predicted_image if necessary
        if district_mask.shape != predicted_image.shape:
            if district_mask.shape[0] > predicted_image.shape[0]:
                district_mask = district_mask[:predicted_image.shape[0], :]
            if district_mask.shape[1] > predicted_image.shape[1]:
                district_mask = district_mask[:, :predicted_image.shape[1]]

        predicted_population_district = np.sum(predicted_image * district_mask)
        true_population_district = ins_population[district]
        true_values_all_districts.append(true_population_district)
        predicted_values_all_districts.append(predicted_population_district)

    mae_district = mean_absolute_error(true_values_all_districts, predicted_values_all_districts)
    print(f"District-level MAE: {mae_district}")
    return mae_district
