import random
import string
import os
import glob
import argparse

def collect_code_files(extensions=('.cpp', '.c', '.h', '.hpp'), 
                      dirs=('src', 'include', 'ggml/src', 'examples', 'tools'),
                      max_length=500000,
                      verbose=True):
    """
    Collect code file contents from repository
    :param extensions: File extensions to collect
    :param dirs: Directories to search
    :param max_length: Maximum character count
    :param verbose: Whether to print detailed information
    :return: Concatenated code text and total character count
    """
    all_content = []
    total_chars = 0
    file_count = 0
    
    # Find all matching file paths
    all_files = []
    for directory in dirs:
        if not os.path.exists(directory):
            if verbose:
                print(f"Directory does not exist: {directory}")
            continue
            
        for ext in extensions:
            pattern = os.path.join(directory, f'**/*{ext}')
            for file_path in glob.glob(pattern, recursive=True):
                all_files.append(file_path)
    
    # Randomly shuffle file order
    random.shuffle(all_files)
    
    # Read file contents
    for file_path in all_files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                file_header = f"\n\n----- FILE: {file_path} -----\n\n"
                
                # Check if adding this file would exceed maximum length limit
                remaining_chars = max_length - total_chars if max_length > 0 else float('inf')
                
                if len(file_header) + len(content) > remaining_chars:
                    # Need to truncate file
                    if remaining_chars > len(file_header) + 100:  # Ensure at least 100 characters of content
                        truncated_content = content[:remaining_chars - len(file_header)]
                        truncated_content += "\n\n... (file content truncated) ..."
                        
                        all_content.append(file_header)
                        all_content.append(truncated_content)
                        total_chars += len(file_header) + len(truncated_content)
                        file_count += 1
                        
                        if verbose:
                            print(f"Added file (truncated): {file_path}, current total chars: {total_chars}, file count: {file_count}")
                        break  # Reached maximum length, exit loop
                    else:
                        # Remaining space too small to add meaningful content
                        if verbose:
                            print(f"Skipped file: {file_path}, insufficient remaining space")
                        continue
                else:
                    # Add file completely
                    all_content.append(file_header)
                    all_content.append(content)
                    total_chars += len(file_header) + len(content)
                    file_count += 1
                    
                    if verbose:
                        print(f"Added file: {file_path}, current total chars: {total_chars}, file count: {file_count}")
                    
                    if max_length > 0 and total_chars >= max_length:
                        break  # Reached maximum length, exit loop
        except Exception as e:
            if verbose:
                print(f"Error reading file {file_path}: {e}")
    
    return ''.join(all_content), total_chars, file_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate long code prompt file')
    parser.add_argument('--output', '-o', type=str, default="prompt.txt", help='Output file name')
    parser.add_argument('--length', '-l', type=int, default=350000, help='Maximum character count')
    parser.add_argument('--question', "-q", type=str, default="\nSummarize the code", help='Question to ask')
    
    args = parser.parse_args()
    
    # Use default values directly
    dirs = ['src', 'cpmcu', 'scripts']
    extensions = ['.cpp', '.c', '.h', '.hpp', '.py']
    
    # Collect code files
    code_content, content_length, file_count = collect_code_files(
        extensions=extensions,
        dirs=dirs,
        max_length=args.length,
    )
    
    # Add question
    final_content = code_content + f"\n\n{args.question}"
    
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(final_content)
    print(f"Generated {args.output}, total length: {len(final_content)} characters, contains {file_count} files")