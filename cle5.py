import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Dict, Any

def clean_and_prepare_dataset(csv_file: str, output_file: str = None) -> pd.DataFrame:
    """
    Clean CSV file by removing problematic rows and saving a cleaned version.
    
    Args:
        csv_file (str): Path to input CSV file
        output_file (str): Path to save cleaned CSV (optional)
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Find rows containing the problematic string
    mask = ~df.apply(lambda x: x.astype(str).str.contains('Quebec Economic Development Program \(QEDP\)', na=False)).any(axis=1)
    
    # Filter out the problematic rows
    cleaned_df = df[mask]
    
    # Save cleaned dataset if output file is specified
    if output_file:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned CSV saved to {output_file}")
        print(f"Removed {len(df) - len(cleaned_df)} problematic rows")
        print(f"Remaining rows: {len(cleaned_df)}")
    
    return cleaned_df

def load_and_prepare_dataset(df: pd.DataFrame) -> DatasetDict:
    """
    Convert cleaned DataFrame to HuggingFace Dataset with train/test splits.
    
    Args:
        df: Pandas DataFrame with cleaned data
        
    Returns:
        datasets.DatasetDict: Prepared dataset with train/test splits
    """
    # Convert DataFrame to Dataset
    dataset = Dataset.from_pandas(df)
    
    # Create train/test splits
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    
    return dataset_dict

def verify_dataset(dataset_dict: DatasetDict) -> Dict[str, Any]:
    """
    Verify the dataset structure and content.
    """
    stats = {
        'num_train': len(dataset_dict['train']),
        'num_test': len(dataset_dict['test']),
        'columns': dataset_dict['train'].column_names,
        'train_sample': dataset_dict['train'][0],
    }
    return stats

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "cleaned_output.csv"
        
        try:
            # Clean the dataset and save to new file
            cleaned_df = clean_and_prepare_dataset(input_file, output_file)
            
            # Convert to HuggingFace dataset
            dataset_dict = load_and_prepare_dataset(cleaned_df)
            
            # Verify the dataset
            stats = verify_dataset(dataset_dict)
            
            print("\nDataset Loading Results:")
            print(f"Training examples: {stats['num_train']}")
            print(f"Test examples: {len(dataset_dict['test'])}")
            print(f"Columns: {', '.join(stats['columns'])}")
            
            print("\nSample from training set:")
            for k, v in stats['train_sample'].items():
                if isinstance(v, str) and len(v) > 100:
                    print(f"{k}: {v[:100]}...")
                else:
                    print(f"{k}: {v}")
                    
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            raise
            
    else:
        print("Usage: python script.py input_csv_file [output_csv_file]")
        sys.exit(1)
