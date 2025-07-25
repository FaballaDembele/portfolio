# Portfolio Budget Analysis Application

## Overview
This project is a web application designed for analyzing government budget data. It provides functionalities for data import, cleaning, exploratory analysis, forecasting, and advanced features such as comparative analysis between institutions.

## Features
- **Homepage**: A welcoming interface with an introduction to the application and example data.
- **Data Import and Cleaning**: Upload CSV files, preview data, and perform cleaning operations.
- **Exploratory Analysis**: Interactive visualizations and key performance indicators (KPIs) to analyze the dataset.
- **Forecasting**: Budget forecasting using LSTM and VAR models with adjustable parameters.
- **Advanced Features**: Comparative analysis between institutions and budget gap analysis.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```
   git clone <repository-url>
   cd portfolio
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the application, execute the following command:
```
streamlit run src/main.py
```

Open your web browser and navigate to `http://localhost:8501` to access the application.

## Project Structure
```
portfolio
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── accueil.py
│   ├── import_clean.py
│   ├── exploratory_analysis.py
│   ├── forecasting.py
│   ├── advanced_features.py
│   ├── utils.py
│   └── styles.py
├── requirements.txt
└── README.md
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.