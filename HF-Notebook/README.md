# Oura Ring Data Visualization and Analysis

This project provides Jupyter notebooks for analyzing and visualizing Oura Ring biometric data alongside hospital flowsheet data, with support for Bland-Altman analysis to compare measurement methods.

## Security and Privacy

This project handles sensitive patient health information (PHI) and requires proper configuration to protect data privacy.

### Sensitive Data Protection

The following data is protected using environment variables:
- Oura Ring API tokens
- Patient Medical Record Numbers (MRNs)
- Patient identifiable information (names, emails, etc.)
- Flowsheet data file paths

**IMPORTANT:** Never commit the `.env` file or data files (`.xlsx`, `.xls`) to version control.

## Setup Instructions

### 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

Or if using conda:

```bash
conda install --file requirements.txt
```

### 2. Configure Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and replace the placeholder values with your actual credentials:
   ```bash
   # Edit with your preferred editor
   nano .env
   # or
   vim .env
   # or
   code .env
   ```

3. Required configuration:
   - `OURA_API_TOKEN`: Your primary Oura Ring API token (get from https://cloud.ouraring.com/personal-access-tokens)
   - `OURA_API_TOKEN_DEMO`: Optional demo token for testing
   - `PATIENT_MRNS`: Comma-separated list of patient MRNs (e.g., `1234567890,9876543210`)
   - `FLOWSHEET_FILE`: Path to your flowsheet Excel file
   - `DEFAULT_TIMEZONE`: Timezone for datetime operations (default: `America/New_York`)

### 3. Prepare Your Data

1. Place your flowsheet Excel file in the project directory
2. Update `FLOWSHEET_FILE` in `.env` to point to your file
3. Ensure your flowsheet contains the required columns:
   - Patient MRN
   - Recorded time/timestamp
   - Measurement values
   - Measurement names
   - Oura API tokens (optional, can use environment variable instead)

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `flowsheet_oura_vis_naomi-Copy1.ipynb` and run the cells.

## Usage

### Patient Selection

The notebook includes an interactive patient selector widget. After configuring your `.env` file with patient MRNs, you can:

1. Select a patient from the dropdown
2. Choose date range for analysis
3. View flowsheet and Oura Ring data for the selected patient

### Data Visualization

The notebook provides visualizations for:

1. **Flowsheet Data**: Interactive plots of all available measurements
2. **Oura Ring Biometrics**:
   - Heart Rate (lowest during sleep)
   - Respiratory Rate
   - Heart Rate Variability (HRV)
   - Sleep Duration
   - Daily Steps
   - Temperature Deviation

3. **Comparative Analysis**: Bland-Altman plots comparing flowsheet vs Oura Ring measurements for:
   - Heart Rate
   - Respiratory Rate
   - Body Temperature

## Configuration Reference

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OURA_API_TOKEN` | Yes | Primary Oura Ring API token |
| `OURA_API_TOKEN_DEMO` | No | Demo token for testing |
| `PATIENT_MRNS` | Yes | Comma-separated patient MRNs |
| `FLOWSHEET_FILE` | Yes | Path to flowsheet data file |
| `DEFAULT_TIMEZONE` | No | Timezone (default: America/New_York) |

### Using config.py

The `config.py` module provides convenient access to environment variables:

```python
from config import Config

# Access configuration values
token = Config.OURA_API_TOKEN
patient_mrns = Config.get_patient_mrns()
file_path = Config.FLOWSHEET_FILE

# Validate configuration
Config.validate()  # Raises ValueError if required vars missing

# Get patient list
patients = Config.get_patient_list_from_env()
```

## Security Best Practices

1. **Never commit sensitive data**:
   - `.env` file is in `.gitignore`
   - Data files (`.xlsx`, `.xls`) are in `.gitignore`
   - Always verify before pushing to remote repositories

2. **Protect API tokens**:
   - Keep tokens in `.env` file only
   - Rotate tokens periodically
   - Use demo tokens for testing when possible

3. **Patient data privacy**:
   - Remove or anonymize patient names in production
   - Use generic identifiers when sharing notebooks
   - Comply with HIPAA and institutional policies

4. **File permissions**:
   ```bash
   # Ensure .env is only readable by you
   chmod 600 .env
   ```

## Troubleshooting

### "Configuration validation failed"

This means required environment variables are missing. Check your `.env` file and ensure all required variables are set.

### "No token found for that MRN"

The flowsheet doesn't contain an Oura API token for the selected patient. The notebook will fall back to using `OURA_API_TOKEN` from `.env`.

### "No rows found for selected patient"

Either:
- The MRN in the flowsheet doesn't match the selected MRN
- The date range doesn't include any data for that patient
- Check MRN format (ensure no extra spaces or formatting)

### Module not found errors

Install missing dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── .env                    # Environment variables (DO NOT COMMIT)
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── config.py              # Configuration loader
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── flowsheet_oura_vis_naomi-Copy1.ipynb  # Main analysis notebook
└── vis.py                # Visualization helper functions (if present)
```

## License and Compliance

This project handles Protected Health Information (PHI). Ensure compliance with:
- HIPAA regulations
- Institutional review board (IRB) requirements
- Data use agreements
- Patient consent requirements

## Support

For issues or questions:
1. Check this README
2. Verify `.env` configuration
3. Check Jupyter notebook error messages
4. Review Oura API documentation: https://cloud.ouraring.com/docs
