

 Walmart Locator & Analytics Suite

A comprehensive Streamlit-based analytics suite for Walmart site selection, demand simulation, and inventory prediction. This project leverages demographic, financial, and competitive data to help identify optimal new store locations, simulate demand shocks, and predict inventory needs.

 Features

- Site Selector AI:  
  Pinpoints high-revenue potential locations for new Walmart stores using grid search, demographic, and financial analysis.
- Demand Shock Simulator:  
  Simulates the impact of demand shocks on candidate locations.
- Inventory Prediction Dashboard:  
  Predicts inventory requirements using advanced time-series and machine learning models.
- Interactive Visualizations:  
  Folium and Plotly-based maps and charts for intuitive exploration of results.

  ![high_level_design Screenshot](models/HIgh_level_design)

 Directory Structure


.
├── main.py                   Streamlit app entry point (Site Selector)
├── integratedapp.py          Integrated multi-app Streamlit interface
├── enhancedintegratedapp.py  Enhanced UI version of the integrated app
├── inventory_app.py          Inventory prediction dashboard
├── analysis_engine.py        Core analysis logic (site selection, demand modeling)
├── explainer_service.py      Generates human-readable explanations for site choices
├── constants.py              Key constants and configuration
├── config.yaml               User-editable configuration (API keys, defaults)
├── requirements.txt          Python dependencies
├── models/
│   └── demand_model.joblib   Trained demand prediction model
├── data/
│   ├── raw/                  Raw input data
│   ├── processed/            Processed/cleaned data
│   └── external/             External data sources
├── utils/
│   ├── data_fetcher.py       Fetches and processes external data (Census, APIs)
│   ├── geo_utils.py          Geographic utilities (FIPS, coordinates)
│   ├── model_utils.py        Model training, loading, and prediction helpers
│   ├── visualization.py      Map and chart visualization utilities
│   ├── config_manager.py     Handles config and API keys
│   └── test_census.py        Utility tests
└── test_census.py            Standalone test script


 Installation

1. Clone the repository:
   bash
   git clone <repo-url>
   cd walmart_locator_rhn
   

2. Install dependencies:
   bash
   pip install -r requirements.txt
   

3. Configure API keys:
   - Edit config.yaml to add your Census and Gemini API keys.

 Usage

 Site Selector App

bash
streamlit run main.py

- Configure parameters in the sidebar (budget, break-even, region, etc.)
- Click "Find New Locations" to run the analysis.
- View candidate sites, demand scores, and detailed explanations.

 Inventory Prediction Dashboard

bash
streamlit run inventory_app.py

- Upload or use provided sales data.
- Explore inventory forecasts, trends, and visualizations.

 Integrated/Enhanced Apps

bash
streamlit run integratedapp.py
 or
streamlit run enhancedintegratedapp.py

- Access all features (site selection, demand simulation, inventory) from a single interface.

 Configuration

- config.yaml:  
  Set analysis defaults, API keys, and visualization options.
- constants.py:  
  Edit or extend supported states, Census variables, and land price ranges.

 Data

- Place raw, processed, or external data in the data/ directory.
- The app can generate synthetic data if real data is missing.

 Dependencies

See requirements.txt for the full list. Key packages:
- streamlit, folium, streamlit-folium, pandas, numpy, scikit-learn, plotly, requests, geopy, joblib, PyYAML

 Extending

- Add new states or regions in constants.py and analysis_engine.py.
- Plug in new models via utils/model_utils.py.
- Customize visualizations in utils/visualization.py.

 License

[MIT License] (or your preferred license)

