import logging
import sys

def setup_logging(level=logging.INFO):
    """Configures basic logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) # Ensure output to stdout
        ]
    )
    # Silence verbose logs from libraries (optional, adjust as needed)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING) 