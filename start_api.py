#!/usr/bin/env python3
"""
Startup script for GuardiaVision FastAPI service
"""

import os
import sys
import subprocess
import argparse

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        from PIL import Image
        from google import genai
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_config():
    """Check if configuration is valid."""
    try:
        # Check if config file exists
        if not os.path.exists("config.json"):
            print("❌ config.json not found")
            return False
        
        # Check if API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY environment variable not set")
            print("Please set it with: export GOOGLE_API_KEY='your-api-key'")
            return False
        
        print("✅ Configuration is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def start_server(host="0.0.0.0", port=8000, workers=1, reload=False):
    """Start the FastAPI server."""
    try:
        cmd = [
            "uvicorn", 
            "app:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers)
        ]
        
        if reload:
            cmd.append("--reload")
        
        print(f"🚀 Starting GuardiaVision API server...")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Workers: {workers}")
        print(f"   Reload: {reload}")
        print(f"   API Docs: http://{host}:{port}/docs")
        print()
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Start GuardiaVision FastAPI service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency and config checks")
    
    args = parser.parse_args()
    
    print("🔍 GuardiaVision FastAPI Startup")
    print("=" * 40)
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check configuration
        if not check_config():
            sys.exit(1)
        
        print()
    
    # Start server
    start_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
