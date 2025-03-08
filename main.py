import subprocess

scripts = ["process_files.py", "summarize.py", "analyze.py", "process_files.py"]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    
    # Print script output
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script}:\n{result.stderr}")
