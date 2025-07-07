#!/usr/bin/env python3
"""
Simple Cross-Node Scheduler for NCSA
Uses file-based locking and status monitoring
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
import fcntl
from datetime import datetime
import socket

class SimpleScheduler:
    def __init__(self, lib: str, low: int = -1, high: int = -1):
        self.lib = lib
        self.low = low
        self.high = high
        self.project_dir = Path(__file__).parent.parent
        
        # Environment variables
        self.apis_per_node = int(os.environ.get('APIS_PER_NODE', '16'))
        self.node_name = socket.gethostname()
        
        # File paths
        self.lock_file = self.project_dir / f".scheduler_lock_{lib}"
        self.monitor_file = self.project_dir / "monitor.txt"
        self.apis_file = self.project_dir / "torch_apis.txt"
        
        # Ensure directories exist
        (self.project_dir / ".tmp").mkdir(exist_ok=True)
        (self.project_dir / "logs").mkdir(exist_ok=True)

    def load_apis(self):
        """Load API list from file or generate from range"""
        if self.apis_file.exists():
            with open(self.apis_file, 'r') as f:
                apis = [line.strip() for line in f if line.strip()]
            return apis
        elif self.low != -1 and self.high != -1:
            return [str(i) for i in range(self.low, self.high + 1)]
        else:
            raise ValueError("No torch_apis.txt file found and no valid range specified")

    def get_next_batch(self, apis):
        """Get next batch of APIs using file locking"""
        while True:
            try:
                # Acquire lock
                with open(self.lock_file, 'a+') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    
                    # Read current status
                    lock.seek(0)
                    content = lock.read()
                    
                    if not content:
                        # Initialize with all APIs as pending
                        processed = set()
                        lock.seek(0)
                        lock.truncate()
                        for api in apis:
                            lock.write(f"{api}:pending\\n")
                    else:
                        # Parse existing status
                        processed = set()
                        for line in content.split('\\n'):
                            if ':' in line:
                                api, status = line.split(':', 1)
                                if status in ['running', 'finished', 'failed']:
                                    processed.add(api)
                    
                    # Find pending APIs
                    pending_apis = [api for api in apis if api not in processed]
                    
                    if not pending_apis:
                        return []  # No more work
                    
                    # Take batch
                    batch = pending_apis[:self.apis_per_node]
                    
                    # Update status to running
                    lock.seek(0)
                    lock.truncate()
                    for api in apis:
                        if api in batch:
                            lock.write(f"{api}:running:{self.node_name}\\n")
                        elif api in processed:
                            # Keep existing status
                            for line in content.split('\\n'):
                                if line.startswith(f"{api}:"):
                                    lock.write(f"{line}\\n")
                                    break
                        else:
                            lock.write(f"{api}:pending\\n")
                    
                    return batch
                    
            except Exception as e:
                print(f"Lock error: {e}")
                time.sleep(1)

    def update_status(self, api, status):
        """Update API status in lock file"""
        try:
            with open(self.lock_file, 'r+') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                
                content = lock.read()
                lines = content.split('\\n')
                
                lock.seek(0)
                lock.truncate()
                
                for line in lines:
                    if line.startswith(f"{api}:"):
                        lock.write(f"{api}:{status}:{self.node_name}\\n")
                    else:
                        lock.write(f"{line}\\n")
                        
        except Exception as e:
            print(f"Status update error: {e}")

    def update_monitor(self, message):
        """Update monitor.txt with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.monitor_file, 'a') as f:
            f.write(f"[{timestamp}] [{self.node_name}] {message}\\n")

    def process_api(self, api):
        """Process single API"""
        self.update_monitor(f"Starting API {api}")
        
        try:
            # Run eval.crash_monitor
            cmd = [
                sys.executable, "-m", "eval.crash_monitor",
                api, self.lib, str(self.low), str(self.high)
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            if result.returncode == 0:
                self.update_status(api, "finished")
                self.update_monitor(f"✓ API {api} finished successfully")
                return True
            else:
                self.update_status(api, "failed")
                self.update_monitor(f"✗ API {api} failed (exit code: {result.returncode})")
                return False
                
        except subprocess.TimeoutExpired:
            self.update_status(api, "failed")
            self.update_monitor(f"⚠ API {api} timed out")
            return False
        except Exception as e:
            self.update_status(api, "failed")
            self.update_monitor(f"✗ API {api} error: {e}")
            return False

    def run(self):
        """Main execution"""
        self.update_monitor(f"Scheduler started for library {self.lib}")
        self.update_monitor(f"APIs per node: {self.apis_per_node}")
        
        # Load APIs
        apis = self.load_apis()
        self.update_monitor(f"Loaded {len(apis)} APIs")
        
        total_processed = 0
        total_success = 0
        
        while True:
            # Get next batch
            batch = self.get_next_batch(apis)
            
            if not batch:
                self.update_monitor("No more APIs to process")
                break
            
            self.update_monitor(f"Processing batch of {len(batch)} APIs: {batch}")
            
            # Process each API in batch
            for api in batch:
                success = self.process_api(api)
                total_processed += 1
                if success:
                    total_success += 1
            
            self.update_monitor(f"Batch completed. Progress: {total_processed} total, {total_success} successful")
        
        self.update_monitor(f"Node finished. Processed: {total_processed}, Successful: {total_success}")

def main():
    parser = argparse.ArgumentParser(description="Simple Cross-Node Scheduler")
    parser.add_argument("lib", choices=["torch", "pytorch", "tf", "tensorflow"], 
                       help="Library to test")
    parser.add_argument("--low", type=int, default=-1, 
                       help="Low range for API generation")
    parser.add_argument("--high", type=int, default=-1, 
                       help="High range for API generation")
    
    args = parser.parse_args()
    
    # Normalize library name
    if args.lib in ["pytorch"]:
        lib = "torch"
    elif args.lib in ["tensorflow"]:
        lib = "tf"
    else:
        lib = args.lib
    
    scheduler = SimpleScheduler(lib=lib, low=args.low, high=args.high)
    scheduler.run()

if __name__ == "__main__":
    main()
