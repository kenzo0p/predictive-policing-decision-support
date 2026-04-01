from data_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline(
        input_path="data/raw/dstrIPC_2013.csv",
        output_path="data/processed/clean_crime_data.csv",
    )
