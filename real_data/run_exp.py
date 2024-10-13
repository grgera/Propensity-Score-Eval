import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from sklearn.metrics import mean_squared_error

from .data_generator import WmtInputGenerator, ParserInputGenerator, QaInputGenerator
from .reweighers import AVAILABLE_REWEIGHERS
from .utils import set_seeds, setup_logging, compute_geom_loss
from tqdm import tqdm


def log_average_improvements(df, logger):
    """
    Reports the average MSE improvement and average EMD improvement for each reweigher independently.

    Parameters:
    df (pd.DataFrame): The DataFrame containing experiment results with columns 'reweigher_name',
                       'mse_imp', and 'emd_imp'.

    Returns:
    pd.DataFrame: A DataFrame with the average improvements per reweigher.
    """
    # Group by reweigher_name and calculate the mean of mse_imp and emd_imp
    improvement_df = df.groupby('reweigher_name').agg(
        base_mse=('base_mse', 'mean'),
        final_mse=('mse', 'mean'),
        base_emd=('base_emd', 'mean'),
        final_emd=('emd', 'mean'),
    ).reset_index()

    # Log or print the results for each reweigher
    for index, row in improvement_df.iterrows():
        logger.info(f"Reweigher: {row['reweigher_name']}, Avg MSE Improvement: {row['base_mse'] - row['final_mse']:.4f}, "
                     f"Avg EMD Improvement: {row['base_emd'] - row['final_emd']:.4f} -- final MSE: {row['final_mse']:.4f} ({row['base_mse']:.4f})")


    return improvement_df


def execute_experiment(setups, reweigher, logger, output_dir, verbose=False):
    """
    Runs the experiment for current setup and logs the results of two success measures.
    Creates and returns a DataFrame with columns: emd, emd_imp, mse, mse_imp, n_samples.
    """
    emd_results = []
    mse_results = []

    emd_improvements = []
    mse_improvements = []

    # Create a list to store data for each setup for the DataFrame
    results_data = []

    for setup_id, setup in tqdm(enumerate(setups), desc="Running each setup"):

        # Train the reweighter
        ## we think that data_a, data_b are list of vectorized texts [num_sampels, 768]
        reweigher.learn_weights(data_a=setup.data_a, data_b=setup.data_b)

        # Apply reweighting to dataset A
        weights_a = reweigher.reweigh(setup.data_a)

        # Measure 1: EMD (Wasserstein distance) between reweighted A and B
        reweighted_emd = compute_geom_loss(x=setup.data_a, y=setup.data_b, x_weights=weights_a)

        # if setup.data_a.shape[1] > 1:
        #     # reweighted_emd = wasserstein_distance_nd(u_values=setup.data_a, v_values=setup.data_b, u_weights=weights_a)
        # else:
        #     reweighted_emd = wasserstein_distance(u_values=setup.data_a.ravel(), v_values=setup.data_b.ravel(), u_weights=weights_a)

        emd_results.append(reweighted_emd)

        # Improvement in EMD
        # emd_imp = setup.emd / reweighted_emd
        # emd_improvements.append(emd_imp)

        # Measure 2: Mean squared error (MSE) between reweighted performance on A and B
        reweighted_perf_a = np.average(setup.performance_a, weights=weights_a)
        true_perf_b = np.mean(setup.performance_b)
        mse = mean_squared_error([true_perf_b], [reweighted_perf_a])
        mse_results.append(mse)

        # Improvement in MSE
        # mse_imp = setup.mse / mse
        # mse_improvements.append(mse_imp)

        # Log details for this setup if verbose
        if verbose:
            logger.info(f"Setup {setup_id + 1}:\n -- EMD after reweighing: {reweighted_emd:.4f} "
                        f"(EMD before reweighing: {setup.emd:.4f}) \t Improvement: {setup.emd-reweighted_emd:.4f}"
                        f"\n -- MSE with reweighing: {mse:.4f} "
                        f"(MSE without reweighing: {setup.mse:.4f}) \t Improvement: {setup.mse - mse:.4f}")

        # Add results for this setup to the list
        results_data.append({
            "emd": reweighted_emd,
            "base_emd": setup.emd,
            "mse": mse,
            "base_mse": setup.mse,
        })

    if verbose:
        logger.info(
            f"Average EMD: {np.mean(emd_results):.4f}, average improvement factor {np.mean(emd_improvements):.4f}")
        logger.info(
            f"Average MSE: {np.mean(mse_results):.4f}, average improvement factor {np.mean(mse_improvements):.4f}")

    # Create a DataFrame from the results_data list
    results_df = pd.DataFrame(results_data)

    return results_df


def experiment_pipeline(config, logger):
    reweighers = config['experiment']['reweighers']
    performance_models = config['experiment']['performance_models']

    final_df_list = []

    # n_setups = config['experiment']['n_setups']
    # n_samples = config['experiment']['n_samples']

    for model_performance_name in performance_models:

        dataset_config = config['dataset']
        dataset_config['model_performance'] = model_performance_name

        input_data_distribution = dataset_config.get('input_data_distribution', '-')

        # N dimensional input distributions
        if input_data_distribution == 'wmt':
            data_generator = WmtInputGenerator(dataset_config)
        elif input_data_distribution == 'parsers':
            data_generator = ParserInputGenerator(dataset_config)
        elif input_data_distribution == 'qa':
            data_generator = QaInputGenerator(dataset_config)
        else:
            raise Exception(f"Input data distribution not supported: {input_data_distribution}")
        
        setups = []
        themes = dataset_config["shifts"]
        source = dataset_config["text_from"]
        for orig_theme in tqdm(themes, desc="Generating each setup"):
            # Generate synthetic data
            for tar_theme in themes:
                if tar_theme != orig_theme:
                    setup = data_generator.get_data(
                        original_theme=orig_theme,
                        target_theme=tar_theme,
                        path_to_data=dataset_config["path_to_data"],
                        source=source,
                        # output_dir=output_dir,
                    )
                    setups.append(setup)

        for reweigher_name in reweighers:
            reweigher = AVAILABLE_REWEIGHERS[reweigher_name]

            if isinstance(reweigher, type):  # Check if it's a class, i.e., needs initialization
                rew_config = config['experiment']["reweighers_configs"][reweigher_name]
                reweigher = reweigher(rew_config)

            logger.info(
                f"Executing reweighing experiment for {model_performance_name} - {reweigher_name} on input {input_data_distribution}")
            # Execute the experiment and get the result DataFrame
            result_df = execute_experiment(
                setups=[setup],
                reweigher=reweigher,
                logger=logger,
                output_dir=output_dir,
            )

            # Add columns to result_df for model_performance_name, reweigher_name, and input_data_distribution
            result_df['model_performance_name'] = model_performance_name
            result_df['reweigher_name'] = reweigher_name
            result_df['input_data_distribution'] = input_data_distribution
            result_df['n_samples'] = len(setup.data_a)

            log_average_improvements(result_df, logger)

            # Append the result DataFrame to the final_df_list
            final_df_list.append(result_df)

    # Concatenate all DataFrames into a single final DataFrame
    final_df = pd.concat(final_df_list, ignore_index=True)

    return final_df


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run meta-model experiment")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    config_file = args.config
    if not config_file.endswith(".yaml"):
        config_file += ".yaml"

    # config_folder = "configs"
    # config_path = os.path.join(config_folder, config_file)

    # Load configuration with OmegaConf
    config = OmegaConf.load(config_file)

    # Add a timestamp to the output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("logs", Path(config_file).stem, f"run_{timestamp}")

    # Setup logging
    logger = setup_logging(output_dir, "output.log")

    # Set all the seeds
    seed = config['experiment'].get("seed", 1234)
    set_seeds(seed)
    logger.info(f"All seeds set to: {seed}")

    result_df = experiment_pipeline(config=config, logger=logger)

    logger.info(f"="*100)
    ## TODO: change for samples number
    # logger.info(f"Average global results (for {result_df.shape[0]} experiments "
    #             f"with {config['experiment']['n_samples']} samples):")
    log_average_improvements(result_df, logger)

    output_path = os.path.join(output_dir, "results.csv")
    result_df.to_csv(output_path, index=False)

    # Dataset generation
    logger.info(f"Execution finished, results logged at {output_dir}")
