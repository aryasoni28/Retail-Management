import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import folium_static
from analysis_engine import AnalysisEngine
from utils.visualization import MapVisualizer
from constants import DEFAULT_CONFIG
from typing import Dict
import plotly.express as px
# Set page config (only once at the top level)
st.set_page_config(
    page_title="Walmart Analytics Suite",
    layout="wide",
    initial_sidebar_state="auto",
    page_icon="🛒"
)

# Custom CSS for navigation
st.markdown("""
<style>
    .nav-container {
        display: flex;
        justify-content: space-around;
        padding: 1rem;
        background-color: #0071CE; /* Walmart Blue */
        margin-bottom: 2rem;
        border-radius: 0.5rem;
    }
    .nav-item {
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        cursor: pointer;
        border-radius: 0.25rem;
        transition: all 0.2s;
    }
    .nav-item:hover {
        background-color: #005FA3;
    }
    .nav-item.active {
        background-color: #FFC220; /* Walmart Yellow */
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# Navigation state
if 'current_app' not in st.session_state:
    st.session_state.current_app = 'Site Selector'

# Navigation bar
col1, col2, col3 = st.columns(3)
with col1:
    if st.button('📍 Site Selector'):
        st.session_state.current_app = 'Site Selector'
with col2:
    if st.button('⚡ Demand Shock Simulator'):
        st.session_state.current_app = 'Demand Shock Simulator'
with col3:
    if st.button('📦 Inventory Prediction'):
        st.session_state.current_app = 'Inventory Prediction'

# Display the selected app
if st.session_state.current_app == 'Site Selector':
    # Import and run site selector app
    def site_selector_app():
        import streamlit as st
        import pandas as pd
        import numpy as np
        from streamlit_folium import folium_static
        from analysis_engine import AnalysisEngine
        from utils.visualization import MapVisualizer
        from constants import DEFAULT_CONFIG
        from typing import Dict
        import plotly.express as px

# --- UI Setup ---
        def setup_page():
            #"""Configures the Streamlit page settings with enhanced styling."""
            #st.set_page_config(
             #   page_title="Walmart Site Selector Pro",
              #  layout="wide",
               # initial_sidebar_state="expanded",
                #page_icon="📍"
            #)
            st.markdown("""
                <style>
                    /* Main content area */
                    .main .block-container {
                        padding-top: 2rem;
                        padding-bottom: 2rem;
                    }
                    
                    /* Cards */
                    .location-card {
                        border-radius: 0.75rem;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                        transition: all 0.3s ease;
                        padding: 1.25rem;
                        margin-bottom: 1rem;
                        background: white;
                        border-left: 4px solid #007dc6;
                    }
                    
                    .location-card:hover {
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        transform: translateY(-3px);
                    }
                    
                    /* Metrics */
                    .metric-card {
                        border-radius: 0.75rem;
                        padding: 1rem;
                        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
                        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    }
                    
                    /* Map container */
                    .map-container {
                        border-radius: 0.75rem;
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        height: 600px;
                    }
                    
                    /* Better tabs */
                    .stTabs [role="tablist"] {
                        gap: 0.5rem;
                    }
                    
                    .stTabs [role="tab"] {
                        border-radius: 0.5rem !important;
                        padding: 0.5rem 1rem !important;
                    }
                    
                    .stTabs [aria-selected="true"] {
                        background-color: #007dc6 !important;
                    }
                </style>
            """, unsafe_allow_html=True)

        def display_sidebar() -> Dict:
            """Renders the sidebar for user input with improved organization."""
            with st.sidebar:
                st.image("walmart.png", width=120)
                st.title("Site Selection AI")
                st.markdown("Configure parameters to find optimal new store locations.")
                
                with st.expander("💰 Financial Parameters", expanded=True):
                    params = {
                        "budget": st.slider(
                            "Max Land Budget ($/sqft)", 5, 300, DEFAULT_CONFIG["analysis"]["budget"], 5,
                            help="Maximum acceptable price per square foot for land acquisition."
                        ),
                        "break_even_months": st.slider(
                            "Target Break-even (months)", 12, 72, DEFAULT_CONFIG["analysis"]["break_even_months"], 6,
                            help="Desired time to recover the initial investment."
                        ),
                        "turnover": st.slider(
                            "Annual Revenue Target ($M)", 5, 150, DEFAULT_CONFIG["analysis"]["target_turnover"],
                            help="Minimum acceptable annual revenue projection."
                        ) * 1_000_000,
                    }

                with st.expander("🗺️ Geographic Parameters", expanded=True):
                    params.update({
                        "search_area_name": st.selectbox(
                            "Search Area",
                            ("Colorado", "Texas"),
                            help="The state or region to analyze."
                        ),
                        "search_radius_miles": st.slider(
                            "Competition Radius (miles)", 1, 50, 15,
                            help="Radius to check for existing competitor stores."
                        ),
                        "max_candidates": 100
                    })

                st.markdown("---")
                if st.button("🚀 Find Optimal Locations", type="primary", use_container_width=True):
                    st.session_state.analysis_params = params
                    st.session_state.run_analysis = True

                return params

        def display_main_content(engine: AnalysisEngine, visualizer: MapVisualizer):
            """Renders the main content area with improved layout."""
            st.title("📍 Walmart Site Selector Pro")
            st.markdown("AI-powered location analysis for optimal new store placement.")
            
            if not st.session_state.get("run_analysis", False):
                st.info("⬅️ Configure parameters in the sidebar and click **'Find Optimal Locations'** to begin.")
                return

            params = st.session_state.analysis_params

            with st.status("Running advanced location analysis...", expanded=True) as status:
                status.write("📍 Generating candidate locations...")
                candidates = engine.generate_candidate_locations(params)

                if candidates.empty:
                    status.update(label="Analysis complete!", state="complete")
                    st.warning(f"No potential sites found within the **${params['budget']}/sqft** budget.")
                    return

                status.write(f"📊 Analyzing {len(candidates)} potential sites...")
                results = engine.analyze_locations(candidates, params)

                if results.empty:
                    status.update(label="Analysis Error!", state="error")
                    st.error("Failed to fetch demographic data. Please try again.")
                    return

                final_results = results[
                    (results["break_even_months"] <= params["break_even_months"]) &
                    (results["break_even_months"] != np.inf)
                ].sort_values("demand_score", ascending=False).reset_index(drop=True)

                st.session_state.results = final_results
                status.update(label="Analysis complete!", state="complete")

            if st.session_state.results.empty:
                st.warning("No locations met all criteria. Consider adjusting financial targets.")
                return

            display_results(st.session_state.results, visualizer, engine, params)

        def display_results(results: pd.DataFrame, visualizer: MapVisualizer, engine: AnalysisEngine, params: Dict):
            """Displays the ranked results with improved layout and visualization."""
            st.success(f"🎯 Found {len(results)} optimal locations matching your criteria")
            
            # Top result section
            top_result = results.iloc[0].to_dict()
            top_result = engine.get_best_location_analysis(top_result)
            
            with st.spinner("Generating AI insights..."):
                explanation = engine.get_location_explanation(top_result)
            
            # Top metrics in cards
            st.subheader("Top Recommendation")
            cols = st.columns(3)
            with cols[0]:
                with st.container(border=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📍 Location", top_result['address'].split(',')[0])
                    st.markdown('</div>', unsafe_allow_html=True)
            with cols[1]:
                with st.container(border=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📈 Demand Score", f"{top_result['demand_score']:.2f}", "0-1 scale")
                    st.markdown('</div>', unsafe_allow_html=True)
            with cols[2]:
                with st.container(border=True):
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💰 Break-Even", f"{top_result['break_even_months']:.1f} months", "Lower is better")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # AI explanation in an expandable card
            with st.expander("📝 AI Location Analysis", expanded=True):
                st.markdown(f"**Why this location was selected:**\n\n{explanation}")
            
            # Main content tabs
            tab1, tab2 = st.tabs(["🗺️ Geographic Analysis", "📊 Top Locations"])
            
            with tab1:
                st.subheader("Geographic Analysis")
                competitors = engine.data_fetcher.get_walmart_locations(params['search_area_name'])
                
                # Create and display the map directly without extra containers
                m = visualizer.create_base_map(results[["lat", "lon"]].values.tolist())
                visualizer.add_walmart_locations(m, competitors)
                visualizer.add_candidate_locations(m, results.head(10))
                folium_static(m, height=550)
                
                # Demographic profile below map
                display_advanced_demographics(top_result)
            
            with tab2:
                st.subheader("Top 5 Recommended Locations")
                
                # Improved location cards with tabs for each
                tabs = st.tabs([f"#{i+1}" for i in range(min(5, len(results)))])
                
                for idx, tab in enumerate(tabs):
                    with tab:
                        row = results.iloc[idx]
                        with st.container():
                            st.markdown('<div class="location-card">', unsafe_allow_html=True)
                            
                            # Header with address and rank
                            st.markdown(f"### {row['address'].split(',')[0]}")
                            
                            # Metrics in columns
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Demand Score", f"{row['demand_score']:.2f}")
                            m2.metric("Break-Even", f"{row['break_even_months']:.1f} mo")
                            m3.metric("Land Cost", f"${int(row['price_per_sqft'])}/sqft")
                            
                            # Detailed info in expander
                            with st.expander("Detailed Analysis"):
                                st.write(f"**Population:** {int(row['population']):,}")
                                st.write(f"**Median Income:** ${int(row['median_income']):,}")
                                st.write(f"**Competition Score:** {row['competition_score']:.2f}")
                                st.write(f"**Coordinates:** `{row['lat']:.4f}, {row['lon']:.4f}`")
                                
                                # Mini map for this location
                                mini_map = visualizer.create_single_location_map(row)
                                folium_static(mini_map, height=300)
                            
                            st.markdown('</div>', unsafe_allow_html=True)

        def display_advanced_demographics(best_location: Dict):
            """Displays enhanced demographic information in an organized way."""
            with st.expander("🧑‍🤝‍🧑 Advanced Demographic Profile", expanded=True):
                # Two column layout for key metrics (removed household size)
                col1, col2 = st.columns(2)
                
                col1.metric(
                    label="📚 Education (10th Grade+)",
                    value=f"{best_location.get('literacy_rate', 'N/A')}%",
                    help="Population with at least 10th grade education"
                )
                
                col2.metric(
                    label="👔 Dominant Occupation",
                    value=best_location.get('major_occupation', 'N/A'),
                    help="Most common employment sector"
                )
                
                # Population distribution visualization with percentages
                st.markdown("### Population Distribution")
                
                population_data = {
                    "Children (0-15)": best_location.get("children_pct", 0),
                    "Working Age (15-65)": best_location.get("working_pct", 0),
                    "Elderly (65+)": best_location.get("elderly_pct", 0)
                }
                
                # Display percentages as text
                st.write(f"""
                - 👶 Children (0-15): {population_data['Children (0-15)']:.1f}%
                - 👨‍💼 Working Age (15-65): {population_data['Working Age (15-65)']:.1f}%
                - 👵 Elderly (65+): {population_data['Elderly (65+)']:.1f}%
                """)
                
                # Use tabs for different visualization options
                tab1, tab2 = st.tabs(["Pie Chart", "Bar Chart"])
                
                with tab1:
                    fig = px.pie(
                        names=list(population_data.keys()),
                        values=list(population_data.values()),
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        hole=0.3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    fig = px.bar(
                        x=list(population_data.keys()),
                        y=list(population_data.values()),
                        color=list(population_data.keys()),
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={'x': 'Age Group', 'y': 'Percentage'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        def main():
            """Main function to run the Streamlit app."""
            setup_page()

            # Initialize singletons
            if 'engine' not in st.session_state:
                st.session_state.engine = AnalysisEngine()
            if 'visualizer' not in st.session_state:
                st.session_state.visualizer = MapVisualizer()

            display_sidebar()
            display_main_content(st.session_state.engine, st.session_state.visualizer)

        if __name__ == "__main__":
            main()

    site_selector_app()

elif st.session_state.current_app == 'Demand Shock Simulator':
    # Import and run demand shock simulator app
    def demand_shock_app():
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Load your CSV
        @st.cache_data
        def load_data():
            return pd.read_csv("walmart_shock_simulated_data.csv")

        df = load_data()

        # 1. Festival impact by state
        state_festival_impact = {
            "California": {
                "Diwali": {"Toys": 0.4, "Electronics": 0.2, "Clothing": 0.3, "Books": 0.15},
                "Lunar New Year": {"Groceries": 0.15, "Beauty": 0.2, "Toys": 0.2},
                "Thanksgiving": {"Groceries": 0.25, "Books": 0.1, "Furniture": 0.1}
            },
            "Texas": {
                "Fiesta San Antonio": {"Toys": 0.3, "Groceries": 0.2, "Books": 0.1},
                "Eid": {"Clothing": 0.25, "Beauty": 0.2, "Groceries": 0.15},
                "4th of July": {"Electronics": 0.2, "Groceries": 0.25, "Toys": 0.1}
            },
            "Wisconsin": {
                "Oktoberfest": {"Groceries": 0.2, "Books": 0.15, "Clothing": 0.1},
                "Thanksgiving": {"Groceries": 0.25, "Books": 0.15, "Furniture": 0.1},
                "Hmong New Year": {"Beauty": 0.2, "Toys": 0.2}
            },
            "Colorado": {
                "Thanksgiving": {"Groceries": 0.25, "Books": 0.15},
                "Cinco de Mayo": {"Toys": 0.3, "Clothing": 0.2},
                "Christmas": {"Electronics": 0.25, "Toys": 0.5}
            }
        }

        # 2. Weather-based impacts
        weather_impact = {
            "Snow": {"Clothing": 0.2, "Books": 0.1, "Electronics": 0.1},
            "Heatwave": {"Beauty": 0.2, "Groceries": 0.1},
            "Rain": {"Books": 0.2, "Groceries": 0.1},
            "None": {}
        }

        # 3. Simulation function
        def simulate_demand_shock(user_input):
            state = user_input["location"]["State"]
            shocks = user_input.get("shocks", {})
            festival = shocks.get("festival", "None")
            weather = shocks.get("weather", "None")
            discount_map = shocks.get("discount", {})

            results = {}

            for product in df["Product_Type"].unique():
                base = df[df["Product_Type"] == product]["Weekly_Units_Sold"].mean()
                adjusted = base

                # Apply festival impact
                fest_impact = state_festival_impact.get(state, {}).get(festival, {}).get(product, 0)
                adjusted *= (1 + fest_impact)

                # Apply weather impact
                weather_impact_val = weather_impact.get(weather, {}).get(product, 0)
                adjusted *= (1 + weather_impact_val)

                # Apply discount impact
                disc = discount_map.get(product, 0)
                disc_impact = 0.015 * (disc * 100) if disc else 0
                adjusted *= (1 + disc_impact)

                results[product] = {
                    "original_prediction": round(base, 2),
                    "adjusted_prediction": round(adjusted, 2),
                    "impact_percentage": round(((adjusted - base) / base) * 100, 2) if base != 0 else 0
                }

            return results

        # Streamlit App
        st.title("Walmart Demand Shock Simulator")
        st.markdown("This app simulates how different shocks (festivals, weather, discounts) affect product demand across states.")

        # Sidebar for inputs
        st.sidebar.header("Simulation Parameters")

        # Location inputs
        st.sidebar.subheader("Location Details")
        state = st.sidebar.selectbox("State", ["California", "Texas", "Wisconsin", "Colorado"])
        demand_score = st.sidebar.slider("Demand Score", 0.0, 1.0, 0.76)
        children_pct = st.sidebar.slider("Children %", 0.0, 100.0, 23.0)
        working_pct = st.sidebar.slider("Working %", 0.0, 100.0, 59.0)
        elder_pct = st.sidebar.slider("Elder %", 0.0, 100.0, 18.0)
        major_occupation = st.sidebar.selectbox("Major Occupation", ["Healthcare", "Education", "Manufacturing", "Technology", "Retail"])
        literacy_rate = st.sidebar.slider("Literacy Rate", 0.0, 100.0, 89.0)

        # Shock inputs
        st.sidebar.subheader("Shock Parameters")
        festival = st.sidebar.selectbox("Festival", ["None", "Diwali", "Lunar New Year", "Thanksgiving", "Fiesta San Antonio", "Eid", "4th of July", "Oktoberfest", "Hmong New Year", "Cinco de Mayo", "Christmas"])
        weather = st.sidebar.selectbox("Weather Condition", ["None", "Snow", "Heatwave", "Rain"])

        # Discount inputs
        st.sidebar.subheader("Discount Parameters")
        st.sidebar.markdown("Enter discount percentages (0-1 scale):")
        discounts = {}
        products = df["Product_Type"].unique()
        for product in products:
            discounts[product] = st.sidebar.slider(f"{product} Discount", 0.0, 1.0, 0.0, 0.05)

        # Prepare user input
        user_input = {
            "location": {
                "State": state,
                "Demand_Score": demand_score,
                "Children_%": children_pct,
                "Working_%": working_pct,
                "Elder_%": elder_pct,
                "Major_Occupation": major_occupation,
                "Literacy_Rate": literacy_rate
            },
            "shocks": {
                "festival": festival,
                "weather": weather,
                "discount": {k: v for k, v in discounts.items() if v > 0}
            }
        }

        # Run simulation
        if st.sidebar.button("Run Simulation"):
            results = simulate_demand_shock(user_input)
            
            # Display results
            st.subheader("Simulation Results")
            
            # Create a DataFrame for display
            result_df = pd.DataFrame.from_dict(results, orient='index')
            result_df.reset_index(inplace=True)
            result_df.columns = ['Product', 'Original Prediction', 'Adjusted Prediction', 'Impact %']
            
            # Display table
            st.dataframe(result_df.style.format({
                'Original Prediction': '{:.0f}',
                'Adjusted Prediction': '{:.0f}',
                'Impact %': '{:.2f}%'
            }))
            
            # Visualization 1: Comparison Bar Chart
            st.subheader("Demand Prediction Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            result_df.plot(x='Product', y=['Original Prediction', 'Adjusted Prediction'], kind='bar', ax=ax)
            plt.ylabel("Units Sold")
            plt.title("Original vs Adjusted Demand Prediction")
            st.pyplot(fig)
            
            # Visualization 2: Impact Percentage
            st.subheader("Demand Impact Percentage")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Product', y='Impact %', data=result_df, ax=ax2)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.title("Percentage Change in Demand Due to Shocks")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            
            # Display user inputs
            st.subheader("Simulation Parameters Used")
            st.json(user_input)
        else:
            st.info("Configure the simulation parameters in the sidebar and click 'Run Simulation' to see results.")

    demand_shock_app()

elif st.session_state.current_app == 'Inventory Prediction':
    # Import and run inventory prediction app
    def inventory_app():
        import streamlit as st
        import pandas as pd
        import numpy as np
        import lightgbm as lgb
        from sklearn.preprocessing import LabelEncoder
        from sklearn.cluster import KMeans
        from sklearn.model_selection import TimeSeriesSplit
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import io
        import os
        from sklearn.metrics import mean_squared_error

        # Custom CSS for better styling
        st.markdown("""
        <style>
            .main > div {
                padding-top: 2rem;
            }
            .stMetric {
                background-color: #000000;
                border: 1px solid #e0e0e0;
                padding: 1rem;
                border-radius: 0.5rem;
            }
        </style>
        """, unsafe_allow_html=True)

        @st.cache_data
        def load_and_process_data(csv_path):
            """Load and preprocess the data"""
            try:
                df = pd.read_csv(csv_path)
                
                # Data preprocessing
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.dayofweek
                df['month'] = df['date'].dt.month
                df['day_of_month'] = df['date'].dt.day
                df['state_id'] = pd.Categorical(df['state_id']).codes

                # Enhanced weather features
                df['temp_bin'] = pd.cut(df['avg_temp'],
                                    bins=[-20, 0, 10, 20, 30, 50],
                                    labels=['extreme_cold', 'cold', 'mild', 'warm', 'hot'])
                df['is_holiday'] = df['event_name_1'].notna().astype(int)

                # Encode categorical variables
                cat_cols = ['dept_name', 'cat_name', 'store_name', 'state_abbr', 'temp_bin']
                encoder = LabelEncoder()
                for col in cat_cols:
                    if df[col].dtype == 'object':
                        df[col] = encoder.fit_transform(df[col])

                # Sort for lag features
                df = df.sort_values(['store_name', 'item_name', 'date'])

                # Historical sales patterns
                for lag in [1, 7, 14, 28]:
                    df[f'lag_{lag}'] = df.groupby(['store_name', 'item_name'])['units_sold'].shift(lag)

                # Rolling statistics
                for window in [7, 14, 28]:
                    df[f'rolling_avg_{window}'] = df.groupby(['store_name', 'item_name'])['units_sold'] \
                                                .transform(lambda x: x.rolling(window).mean())

                # Income and lifestyle features
                df['is_premium'] = ((df['income_high'] == 1) & (df['prefers_premium'] == 1)).astype(int)
                df['is_value'] = ((df['income_low'] == 1) & (df['prefers_value'] == 1)).astype(int)

                # Store clusters
                store_features = df.groupby('store_name')[['income_high', 'income_mid', 'avg_temp']].mean()
                df['store_cluster'] = df['store_name'].map(dict(zip(store_features.index,
                                                                KMeans(n_clusters=3).fit_predict(store_features))))

                # Seasonality
                seasons = [1]*3 + [2]*3 + [3]*3 + [4]*3
                month_to_season = dict(zip(range(1,13), seasons))
                df['season'] = df['month'].map(month_to_season)

                return df, month_to_season
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return None, None

        @st.cache_resource
        def train_model(df):
            """Train the LightGBM model"""
            try:
                # Remove initial rows without lag data
                df_clean = df.dropna(subset=['lag_28'])
                
                # Prepare features
                exclude_cols = ['units_sold', 'date', 'id', 'item_name', 'event_name_1',
                            'event_type_1', 'state', 'd']
                X = df_clean.drop([col for col in exclude_cols if col in df_clean.columns], axis=1)
                y = df_clean['units_sold']

                # Categorical features for LightGBM
                categorical_features = ['state_id', 'dept_name', 'cat_name', 'store_name',
                                    'state_abbr', 'temp_bin', 'season', 'store_cluster']

                # Train model
                model = lgb.LGBMRegressor(
                    objective='poisson',
                    boosting_type='dart',
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=42,
                    categorical_feature=categorical_features
                )

                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for faster training
                rmse_scores = []
                
                for fold, (train_index, test_index) in enumerate(tscv.split(X)):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    model.fit(X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                            categorical_feature=categorical_features)

                    preds = model.predict(X_test)
                    rmse = np.sqrt(mean_squared_error(y_test, preds))
                    rmse_scores.append(rmse)

                return model, X.columns, np.mean(rmse_scores), categorical_features
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                return None, None, None, None

        def predict_inventory(df, model, feature_columns, store_location, target_month, month_to_season):
            """Generate inventory predictions"""
            try:
                # Filter reference data for the selected location
                location_data = df[df['state_abbr'] == store_location].copy()
                
                if location_data.empty:
                    st.error(f"No data found for location: {store_location}")
                    return None

                # Create prediction dataset
                prediction_data = location_data.groupby(['item_name', 'cat_name', 'dept_name']).last().reset_index()

                # Set temporal features for prediction month
                prediction_data['month'] = target_month
                prediction_data['season'] = month_to_season[target_month]

                # Prepare features
                X_pred = prediction_data[feature_columns]

                # Make predictions
                prediction_data['predicted_units'] = model.predict(X_pred)

                return prediction_data
            except Exception as e:
                st.error(f"Error making predictions: {str(e)}")
                return None

        def create_visualizations(prediction_data, store_location, target_month):
            """Create interactive visualizations using Plotly"""
            
            # 1. Top Products Chart
            top_products = prediction_data.nlargest(15, 'predicted_units')
            fig1 = px.bar(
                top_products, 
                y='item_name', 
                x='predicted_units',
                orientation='h',
                title=f'Top 15 Products to Stock - {store_location} (Month {target_month})',
                labels={'predicted_units': 'Predicted Demand', 'item_name': 'Product'}
            )
            fig1.update_layout(height=600)
            
            # 2. Category Breakdown
            category_demand = prediction_data.groupby('cat_name')['predicted_units'].sum().reset_index()
            fig2 = px.pie(
                category_demand, 
                values='predicted_units', 
                names='cat_name',
                title=f'Demand by Category - {store_location}'
            )
            
            # 3. Department Analysis
            dept_demand = prediction_data.groupby('dept_name')['predicted_units'].sum().reset_index()
            fig3 = px.bar(
                dept_demand, 
                x='dept_name', 
                y='predicted_units',
                title=f'Demand by Department - {store_location}',
                labels={'predicted_units': 'Predicted Demand', 'dept_name': 'Department'}
            )
            
            return fig1, fig2, fig3

        def main():
            st.title("📦 Inventory Prediction Dashboard")
            st.markdown("Generate data-driven inventory recommendations using machine learning")

            # Sidebar for file upload and parameters
            st.sidebar.header("Configuration")
            
            # File upload
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV file", 
                type=['csv'],
                help="Upload your M5 dataset CSV file"
            )
            
            if uploaded_file is not None:
                # Load and process data
                with st.spinner("Loading and processing data..."):
                    df, month_to_season = load_and_process_data(uploaded_file)
                
                if df is not None:
                    # Display data info
                    st.sidebar.success(f"Data loaded: {len(df)} rows")
                    
                    # Train model
                    with st.spinner("Training model..."):
                        model, feature_columns, avg_rmse, categorical_features = train_model(df)
                    
                    if model is not None:
                        st.sidebar.success(f"Model trained (RMSE: {avg_rmse:.2f})")
                        
                        # User inputs
                        st.sidebar.subheader("Prediction Parameters")
                        
                        # Get unique locations
                        unique_locations = df['state_abbr'].unique()
                        store_location = st.sidebar.selectbox(
                            "Select Store Location:",
                            unique_locations
                        )
                        
                        target_month = st.sidebar.slider(
                            "Target Month:",
                            min_value=1,
                            max_value=12,
                            value=6,
                            help="Select the month for inventory prediction"
                        )
                        
                        # Generate predictions button
                        if st.sidebar.button("Generate Predictions", type="primary"):
                            with st.spinner("Generating predictions..."):
                                prediction_data = predict_inventory(
                                    df, model, feature_columns, store_location, target_month, month_to_season
                                )
                            
                            if prediction_data is not None:
                                # Create tabs for different views
                                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Visualizations", "📋 Detailed Report", "💾 Download"])
                                
                                with tab1:
                                    st.header("Prediction Overview")
                                    
                                    # Key metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        total_demand = prediction_data['predicted_units'].sum()
                                        st.metric("Total Predicted Demand", f"{total_demand:,.0f}")
                                    
                                    with col2:
                                        num_products = len(prediction_data)
                                        st.metric("Number of Products", num_products)
                                    
                                    with col3:
                                        avg_demand = prediction_data['predicted_units'].mean()
                                        st.metric("Average Demand per Product", f"{avg_demand:.1f}")
                                    
                                    with col4:
                                        top_product_demand = prediction_data['predicted_units'].max()
                                        st.metric("Highest Product Demand", f"{top_product_demand:.0f}")
                                    
                                    # Top 10 products table
                                    st.subheader("Top 10 Products")
                                    top_10 = prediction_data.nlargest(10, 'predicted_units')[
                                        ['item_name', 'cat_name', 'dept_name', 'predicted_units']
                                    ]
                                    st.dataframe(top_10, use_container_width=True)
                                
                                with tab2:
                                    st.header("Visualizations")
                                    
                                    # Create visualizations
                                    fig1, fig2, fig3 = create_visualizations(prediction_data, store_location, target_month)
                                    
                                    # Display charts
                                    st.plotly_chart(fig1, use_container_width=True)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.plotly_chart(fig2, use_container_width=True)
                                    with col2:
                                        st.plotly_chart(fig3, use_container_width=True)
                                
                                with tab3:
                                    st.header("Detailed Inventory Report")
                                    
                                    # Prepare detailed report
                                    report = prediction_data[['item_name', 'cat_name', 'dept_name', 'predicted_units']].copy()
                                    report = report.sort_values('predicted_units', ascending=False)
                                    report.columns = ['Product', 'Category', 'Department', 'Recommended Stock']
                                    
                                    # Add safety stock (20% buffer)
                                    report['Safety Stock'] = (report['Recommended Stock'] * 1.2).round().astype(int)
                                    report['Recommended Stock'] = report['Recommended Stock'].round().astype(int)
                                    
                                    # Display with filtering options
                                    st.subheader("Filter Options")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        selected_category = st.selectbox(
                                            "Filter by Category:",
                                            ['All'] + list(report['Category'].unique())
                                        )
                                    
                                    with col2:
                                        selected_department = st.selectbox(
                                            "Filter by Department:",
                                            ['All'] + list(report['Department'].unique())
                                        )
                                    
                                    # Apply filters
                                    filtered_report = report.copy()
                                    if selected_category != 'All':
                                        filtered_report = filtered_report[filtered_report['Category'] == selected_category]
                                    if selected_department != 'All':
                                        filtered_report = filtered_report[filtered_report['Department'] == selected_department]
                                    
                                    st.dataframe(filtered_report, use_container_width=True)
                                
                                with tab4:
                                    st.header("Download Reports")
                                    
                                    # Prepare Excel download
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        report.to_excel(writer, sheet_name='Inventory_Recommendations', index=False)
                                        
                                        # Add summary sheet
                                        summary_data = {
                                            'Metric': ['Total Predicted Demand', 'Number of Products', 'Average Demand', 'Store Location', 'Target Month'],
                                            'Value': [total_demand, num_products, f"{avg_demand:.1f}", store_location, target_month]
                                        }
                                        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                                    
                                    output.seek(0)
                                    
                                    st.download_button(
                                        label="📥 Download Excel Report",
                                        data=output.getvalue(),
                                        file_name=f"inventory_recommendations_{store_location}_{target_month}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                    
                                    # CSV download
                                    csv_data = report.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download CSV Report",
                                        data=csv_data,
                                        file_name=f"inventory_recommendations_{store_location}_{target_month}.csv",
                                        mime="text/csv"
                                    )
                        
                        # Model information
                        st.sidebar.subheader("Model Information")
                        st.sidebar.info(f"""
                **Model**: LightGBM Regressor
                **Objective**: Poisson
                **Features**: {len(feature_columns)}
                **Cross-validation RMSE**: {avg_rmse:.2f}
                """)
    
            else:
                st.info("👆 Please upload a CSV file to get started")
                st.markdown("""
        ### Expected Data Format
        Your CSV file should contain the following columns:
        - `date`: Date of the record
        - `units_sold`: Number of units sold
        - `item_name`: Product name
        - `store_name`: Store identifier
        - `state_abbr`: State abbreviation
        - `dept_name`: Department name
        - `cat_name`: Category name
        - `avg_temp`: Average temperature
        - `income_high`, `income_mid`, `income_low`: Income demographics
        - `prefers_premium`, `prefers_value`: Customer preferences
        - `event_name_1`: Holiday/event information
        """)
        
        # Call the main function
        main()

    inventory_app()