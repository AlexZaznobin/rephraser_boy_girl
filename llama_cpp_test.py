import os
from huggingface_hub import hf_hub_download, list_repo_files
from llama_cpp import Llama

# Function to download model files
def download_model_files(repo_id, quantization, local_dir):
    """
    Downloads the specified quantized model files from Hugging Face.

    Args:
        repo_id (str): Repository ID on Hugging Face.
        quantization (str): Target quantization level (e.g., "Q8_0").
        local_dir (str): Local directory to save the files.

    Returns:
        str: Path to the first part of the model file.
    """
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # List all files in the repository
    print("Fetching file list from repository...")
    repo_files = list_repo_files(repo_id)

    # Filter files matching the quantization level
    matching_files = [
        file for file in repo_files if quantization in file and file.endswith(".gguf")
    ]

    if not matching_files:
        raise FileNotFoundError(f"No files matching quantization level '{quantization}' found in {repo_id}.")

    # Download all matching files
    downloaded_files = []
    for file in matching_files:
        print(f"Downloading: {file}")
        local_path = hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir)
        downloaded_files.append(local_path)

    # Ensure at least one file was downloaded
    if not downloaded_files:
        raise FileNotFoundError(f"No quantized files for '{quantization}' could be downloaded.")

    # Return the first part of the quantized model
    model_path = next((file for file in downloaded_files if "-00001-of-" in file), None)
    if not model_path:
        raise ValueError(f"Could not find the first part of the model for quantization '{quantization}'.")
        model_path=downloaded_files[0]
    print(f"Model files downloaded successfully. Using: {model_path}")
    return model_path

# Function to load the model and generate a response
def use_model(model_path, prompt):
    """
    Loads the model and generates a response for the given prompt.

    Args:
        model_path (str): Path to the first part of the model file.
        repo_id (str): Repository ID on Hugging Face.
        prompt (str): User prompt for the chat model.

    Returns:
        str: Generated response from the model.
    """
    # Load the model
    print(f"Loading model from: {model_path}")
    # llm = Llama(
    #     model_path=model_path,
    #     verbose=False
    # )
    llm = Llama(
        model_path="./models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf",
        n_gpu_layers=-1, # Uncomment to use GPU acceleration
        # verbose=False
    )

    print("llm loaded ")
    # Create a chat completion
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Extract the response text
    return response["choices"][0]["message"]["content"]

# Main script
if __name__ == '__main__':
    # Define repository and parameters
    REPO_ID = "bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF"
    QUANTIZATION = "Q2_K"  # Target quantization level
    LOCAL_DIR = "./models"
    PROMPT = "Продолжи - говорят, что курица не птица , но с куриными мозгами хватишь горя ?"

    # Download model files
    # model_file = download_model_files(REPO_ID, QUANTIZATION, LOCAL_DIR)
    model_file="""/home/ubuntu/rephraser_boy_girl/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q2_K.gguf"""
    # Use the model to generate a response
    response = use_model(model_file,  PROMPT)

    # Print the response
    print("Response:", response)
