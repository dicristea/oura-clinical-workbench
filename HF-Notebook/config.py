"""
Configuration module for loading environment variables.
This module handles loading sensitive data from .env file for use in Jupyter notebooks.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load .env file from the same directory as this config file
ENV_PATH = Path(__file__).parent / '.env'

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)


class Config:
    """Configuration class for accessing environment variables."""

    # Oura API Configuration
    OURA_API_TOKEN: str = os.getenv('OURA_API_TOKEN', '')
    OURA_API_TOKEN_DEMO: str = os.getenv('OURA_API_TOKEN_DEMO', '')

    # Patient Data Configuration
    @staticmethod
    def get_patient_mrns() -> List[str]:
        """
        Get list of patient MRNs from environment variable.
        Returns a list of MRN strings.
        """
        mrns_str = os.getenv('PATIENT_MRNS', '')
        if not mrns_str:
            return []
        return [mrn.strip() for mrn in mrns_str.split(',') if mrn.strip()]

    # File Paths
    FLOWSHEET_FILE: str = os.getenv('FLOWSHEET_FILE', 'oura_flowsheet_sample.xlsx')

    # Timezone Configuration
    DEFAULT_TIMEZONE: str = os.getenv('DEFAULT_TIMEZONE', 'America/New_York')

    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required environment variables are set.
        Returns True if valid, raises ValueError if not.
        """
        errors = []

        if not cls.OURA_API_TOKEN:
            errors.append("OURA_API_TOKEN is not set")

        if not cls.get_patient_mrns():
            errors.append("PATIENT_MRNS is not set or empty")

        if not cls.FLOWSHEET_FILE:
            errors.append("FLOWSHEET_FILE is not set")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" +
                "\n".join(f"  - {error}" for error in errors) +
                "\n\nPlease check your .env file."
            )

        return True

    @classmethod
    def get_patient_list_from_env(cls) -> List[dict]:
        """
        Create a patient list structure from environment variables.
        Returns a list of patient dictionaries compatible with the notebook format.
        """
        mrns = cls.get_patient_mrns()
        return [
            {
                "mrn": mrn,
                "name": f"Patient {i+1}",  # Generic name
                "notes": "Loaded from environment"
            }
            for i, mrn in enumerate(mrns)
        ]


# Convenience function for notebooks
def load_config() -> Config:
    """
    Load and validate configuration.
    Returns Config instance if valid, raises ValueError if not.
    """
    Config.validate()
    return Config


# Check if .env file exists
def check_env_file() -> bool:
    """Check if .env file exists and provide helpful message if not."""
    if not ENV_PATH.exists():
        print(f"⚠️  WARNING: .env file not found at {ENV_PATH}")
        print("Please create a .env file based on .env.example")
        print(f"You can copy .env.example to .env and fill in your actual values.")
        return False
    return True


# Auto-check on import
if not check_env_file():
    print("\nTo create your .env file:")
    print(f"  1. Copy .env.example to .env")
    print(f"  2. Edit .env and replace placeholder values with your actual credentials")
    print(f"  3. Restart your notebook kernel")
