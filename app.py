import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
# Set page config
st.set_page_config(
    page_title="European Banking Churn Analytics Portal",
    page_icon="🇪🇺",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom CSS styling for premium look & feel (dark/blue premium theme)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Title and Header style */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #38bdf8;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 0px 4px 12px rgba(56, 189, 248, 0.15);
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h2 {
        color: #38bdf8 !important;
    }
    
    /* Card style */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 8px 30px 0 rgba(56, 189, 248, 0.15);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .metric-desc {
        font-size: 0.8rem;
        color: #64748b;
    }
    
    /* Alerts and custom text styles */
    .highlight-red { color: #f43f5e; font-weight: 600; }
    .highlight-green { color: #10b981; font-weight: 600; }
    .highlight-blue { color: #0ea5e9; font-weight: 600; }
    
    /* Segment table style */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 0.9em;
        font-family: sans-serif;
        min-width: 400px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8;
        background-color: #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #38bdf8 !important;
        color: #0f172a !important;
        border-color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)
# Helper function to load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    file_path = "churn_modelling.csv"
    if not os.path.exists(file_path):
        # Fallback if file doesn't exist, create it
        from generate_data import generate_banking_dataset
        generate_banking_dataset(file_path)
        
    df = pd.read_csv(file_path)
    
    # 1. Age Segmentation
    # <30, 30–45, 46–60, 60+
    conditions = [
        (df['Age'] < 30),
        (df['Age'] >= 30) & (df['Age'] <= 45),
        (df['Age'] > 45) & (df['Age'] <= 60),
        (df['Age'] > 60)
    ]
    age_choices = ['<30', '30–45', '46–60', '60+']
    df['AgeSegment'] = np.select(conditions, age_choices, default='Unknown')
    
    # 2. Credit Score Bands
    # Low (<600), Medium (600-700), High (>700)
    conditions = [
        (df['CreditScore'] < 600),
        (df['CreditScore'] >= 600) & (df['CreditScore'] <= 700),
        (df['CreditScore'] > 700)
    ]
    credit_choices = ['Low (<600)', 'Medium (600-700)', 'High (>700)']
    df['CreditBand'] = np.select(conditions, credit_choices, default='Unknown')
    
    # 3. Tenure Groups
    # New (0-2 years), Mid-term (3-7 years), Long-term (8+ years)
    conditions = [
        (df['Tenure'] <= 2),
        (df['Tenure'] >= 3) & (df['Tenure'] <= 7),
        (df['Tenure'] >= 8)
    ]
    tenure_choices = ['New (0-2 years)', 'Mid-term (3-7 years)', 'Long-term (8+ years)']
    df['TenureGroup'] = np.select(conditions, tenure_choices, default='Unknown')
    
    # 4. Balance Segments
    # Zero-balance (=0), Low-balance (>0 and <100k), High-balance (>=100k)
    conditions = [
        (df['Balance'] == 0),
        (df['Balance'] > 0) & (df['Balance'] < 100000),
        (df['Balance'] >= 100000)
    ]
    balance_choices = ['Zero-balance', 'Low-balance (<€100k)', 'High-balance (>=€100k)']
    df['BalanceSegment'] = np.select(conditions, balance_choices, default='Unknown')
    
    # Label descriptions for display
    df['Status'] = df['Exited'].map({0: 'Retained', 1: 'Churned'})
    df['ActivityStatus'] = df['IsActiveMember'].map({0: 'Inactive', 1: 'Active'})
    df['CreditCardStatus'] = df['HasCrCard'].map({0: 'No Card', 1: 'Has Card'})
    
    return df
# Load the data
df_raw = load_and_preprocess_data()
# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/Flag_of_Europe.svg", width=80)
st.sidebar.markdown("## Analytical Controls")
st.sidebar.write("Filter the customer base below:")
# Filter values
geographies = ['All'] + sorted(df_raw['Geography'].unique().tolist())
selected_geo = st.sidebar.selectbox("Geography", geographies)
genders = ['All'] + sorted(df_raw['Gender'].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", genders)
credit_bands = ['All'] + ['Low (<600)', 'Medium (600-700)', 'High (>700)']
selected_credit = st.sidebar.selectbox("Credit Score Band", credit_bands)
activity_statuses = ['All'] + ['Active', 'Inactive']
selected_activity = st.sidebar.selectbox("Activity Status", activity_statuses)
# Age Slider Filter
min_age, max_age = int(df_raw['Age'].min()), int(df_raw['Age'].max())
selected_age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
# Apply filters to working dataframe
df = df_raw.copy()
if selected_geo != 'All':
    df = df[df['Geography'] == selected_geo]
if selected_gender != 'All':
    df = df[df['Gender'] == selected_gender]
if selected_credit != 'All':
    df = df[df['CreditBand'] == selected_credit]
if selected_activity != 'All':
    df = df[df['ActivityStatus'] == selected_activity]
df = df[(df['Age'] >= selected_age_range[0]) & (df['Age'] <= selected_age_range[1])]
# ----------------- MAIN TITLE & HEADER -----------------
st.markdown('<div class="main-title">European Banking Churn Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Policy Dashboard & Customer Segmentation Framework | Mentored by the European Central Bank</div>', unsafe_allow_html=True)
# ----------------- CALCULATE CORE KPIs -----------------
total_customers = len(df)
if total_customers > 0:
    churn_count = int(df['Exited'].sum())
    overall_churn_rate = churn_count / total_customers
    active_count = int(df['IsActiveMember'].sum())
    active_member_pct = active_count / total_customers
    total_balance_at_risk = df[df['Exited'] == 1]['Balance'].sum()
    retained_balance = df[df['Exited'] == 0]['Balance'].sum()
    
    # High Value Churn (Balance >= 100k)
    high_value_customers = df[df['Balance'] >= 100000]
    high_val_total = len(high_value_customers)
    if high_val_total > 0:
        high_val_churn_count = int(high_value_customers['Exited'].sum())
        high_val_churn_rate = high_val_churn_count / high_val_total
    else:
        high_val_churn_rate = 0.0
else:
    overall_churn_rate = 0.0
    active_member_pct = 0.0
    total_balance_at_risk = 0.0
    high_val_churn_rate = 0.0
# ----------------- TABS CREATION -----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary & KPIs",
    "🌍 Regional & Demographic Risk",
    "💎 High-Value Customer Analytics",
    "🧬 Dynamic Segmentation",
    "🤖 Interactive Churn Predictor",
    "📜 Policy Recommendations"
])
# ----------------- TAB 1: EXECUTIVE SUMMARY -----------------
with tab1:
    st.markdown("### Portfoliowide Key Performance Indicators")
    
    # KPI Grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Churn Rate</div>
            <div class="metric-value highlight-red">{overall_churn_rate*100:.2f}%</div>
            <div class="metric-desc">Target Rate: &lt; 10%<br><b>{churn_count:,}</b> churned / <b>{total_customers:,}</b> active</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High-Value Churn Ratio</div>
            <div class="metric-value highlight-red">{high_val_churn_rate*100:.2f}%</div>
            <div class="metric-desc">Segment: Account Balance &ge; €100k<br>Premium customer churn exposure</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Balance at Risk</div>
            <div class="metric-value highlight-blue">€{total_balance_at_risk/1e6:.2f}M</div>
            <div class="metric-desc">Capital flight from churned accounts<br>Retained Capital: <b>€{retained_balance/1e6:.1f}M</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Engagement Index</div>
            <div class="metric-value highlight-green">{active_member_pct*100:.2f}%</div>
            <div class="metric-desc">Active members in selected slice<br>Inactive base: <b>{total_customers - active_count:,}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    if total_customers == 0:
        st.warning("No customers match the current filter selection in the sidebar. Please adjust your filters.")
    else:
        st.markdown("---")
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.markdown("#### Customer Retention Status (Volume Split)")
            fig_pie = px.pie(
                df, 
                names='Status', 
                color='Status',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
                hole=0.4,
                height=350
            )
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(font=dict(color='#f8fafc')),
                font=dict(color='#f8fafc')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.markdown("#### Geographic Risk Index (Total Customer Distribution)")
            geo_df = df.groupby(['Geography', 'Status']).size().reset_index(name='Count')
            fig_geo_bar = px.bar(
                geo_df,
                x='Geography',
                y='Count',
                color='Status',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
                barmode='stack',
                height=350
            )
            fig_geo_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='Region', color='#f8fafc', gridcolor='#334155'),
                yaxis=dict(title='Customer Count', color='#f8fafc', gridcolor='#334155'),
                legend=dict(font=dict(color='#f8fafc')),
                font=dict(color='#f8fafc')
            )
            st.plotly_chart(fig_geo_bar, use_container_width=True)
# ----------------- TAB 2: REGIONAL & DEMOGRAPHIC RISK -----------------
with tab2:
    st.markdown("### Regional & Demographic Risk Profile")
    
    col_geo_left, col_geo_right = st.columns(2)
    
    with col_geo_left:
        st.markdown("#### Churn Rate by Geography")
        geo_churn = df.groupby('Geography')['Exited'].mean().reset_index()
        geo_churn['Churn Rate (%)'] = (geo_churn['Exited'] * 100).round(2)
        
        fig_geo = px.bar(
            geo_churn,
            x='Geography',
            y='Churn Rate (%)',
            text='Churn Rate (%)',
            color='Geography',
            color_discrete_sequence=['#38bdf8', '#fbbf24', '#f87171'],
            height=350
        )
        fig_geo.update_traces(textposition='outside')
        fig_geo.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Country', color='#f8fafc', gridcolor='#334155'),
            yaxis=dict(title='Churn Rate (%)', color='#f8fafc', gridcolor='#334155', range=[0, max(geo_churn['Churn Rate (%)'].max()*1.2, 10)]),
            showlegend=False,
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_geo, use_container_width=True)
        st.caption("🚨 **Insight**: German customers churn at double the rate of French and Spanish customers, pointing to product-market mismatch or aggressive local competitors.")
        
    with col_geo_right:
        st.markdown("#### Engagement Drop Indicator (Activity vs Churn)")
        act_churn = df.groupby(['ActivityStatus', 'Status']).size().reset_index(name='Count')
        
        fig_act = px.bar(
            act_churn,
            x='ActivityStatus',
            y='Count',
            color='Status',
            color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
            barmode='group',
            height=350
        )
        fig_act.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Member Status', color='#f8fafc', gridcolor='#334155'),
            yaxis=dict(title='Customer Count', color='#f8fafc', gridcolor='#334155'),
            legend=dict(font=dict(color='#f8fafc')),
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_act, use_container_width=True)
        st.caption("💡 **Insight**: Inactive members are far more prone to churn. Enhancing digital app features and contact campaigns are critical prevention steps.")
    st.markdown("---")
    col_demo_left, col_demo_right = st.columns(2)
    
    with col_demo_left:
        st.markdown("#### Age Segment Churn Exposure")
        # Ensure correct ordering of Age segments
        age_order = ['<30', '30–45', '46–60', '60+']
        age_churn = df.groupby('AgeSegment')['Exited'].agg(['mean', 'count']).reset_index()
        age_churn['Churn Rate (%)'] = (age_churn['mean'] * 100).round(2)
        # Reorder
        age_churn['AgeSegment'] = pd.Categorical(age_churn['AgeSegment'], categories=age_order, ordered=True)
        age_churn = age_churn.sort_values('AgeSegment')
        
        fig_age = px.line(
            age_churn,
            x='AgeSegment',
            y='Churn Rate (%)',
            markers=True,
            text='Churn Rate (%)',
            height=350
        )
        fig_age.update_traces(textposition='top center', line_color='#38bdf8', marker=dict(size=10, color='#0ea5e9'))
        fig_age.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Age Bracket', color='#f8fafc', gridcolor='#334155'),
            yaxis=dict(title='Churn Rate (%)', color='#f8fafc', gridcolor='#334155', range=[0, max(age_churn['Churn Rate (%)'].max()*1.2, 10)]),
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("⚠️ **Critical Bracket**: Customers aged **46–60 (Mid-Career)** show extremely high risk. This cohort likely has higher asset values and is highly targeted by competitors.")
        
    with col_demo_right:
        st.markdown("#### Tenure Group Churn Comparison")
        ten_order = ['New (0-2 years)', 'Mid-term (3-7 years)', 'Long-term (8+ years)']
        ten_churn = df.groupby('TenureGroup')['Exited'].agg(['mean', 'count']).reset_index()
        ten_churn['Churn Rate (%)'] = (ten_churn['mean'] * 100).round(2)
        ten_churn['TenureGroup'] = pd.Categorical(ten_churn['TenureGroup'], categories=ten_order, ordered=True)
        ten_churn = ten_churn.sort_values('TenureGroup')
        
        fig_ten = px.bar(
            ten_churn,
            x='TenureGroup',
            y='Churn Rate (%)',
            text='Churn Rate (%)',
            color='TenureGroup',
            color_discrete_sequence=['#93c5fd', '#60a5fa', '#2563eb'],
            height=350
        )
        fig_ten.update_traces(textposition='outside')
        fig_ten.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Tenure Segment', color='#f8fafc', gridcolor='#334155'),
            yaxis=dict(title='Churn Rate (%)', color='#f8fafc', gridcolor='#334155', range=[0, max(ten_churn['Churn Rate (%)'].max()*1.2, 10)]),
            showlegend=False,
            font=dict(color='#f8fafc')
        )
        st.plotly_chart(fig_ten, use_container_width=True)
        st.caption("📊 **Tenure Stability**: Churn remains relatively stable across different tenure tiers, showing that tenure alone is not a primary driver of flight.")
# ----------------- TAB 3: HIGH-VALUE CUSTOMERS -----------------
with tab3:
    st.markdown("### High-Value Customer Churn Analysis")
    st.markdown("Understanding churn patterns among premium account holders (Balance ≥ €100,000) is critical, as they represent the core deposit stability of the retail banking system.")
    
    # Sub-filter data for High-Value
    hvc_df = df[df['Balance'] >= 100000]
    total_hvc = len(hvc_df)
    
    if total_hvc == 0:
        st.info("No High-Value customers (Balance >= €100k) match the current filter selection.")
    else:
        hvc_churn_rate = hvc_df['Exited'].mean()
        hvc_churn_count = hvc_df['Exited'].sum()
        hvc_capital_risk = hvc_df[hvc_df['Exited'] == 1]['Balance'].sum()
        
        col_hv1, col_hv2, col_hv3 = st.columns(3)
        with col_hv1:
            st.metric("Total High-Value Customers", f"{total_hvc:,}", f"{total_hvc/total_customers*100:.1f}% of total base")
        with col_hv2:
            st.metric("High-Value Churn Rate", f"{hvc_churn_rate*100:.2f}%", f"{hvc_churn_rate - overall_churn_rate:+.2f}% vs Average", delta_color="inverse")
        with col_hv3:
            st.metric("Capital Flight Risk (HVC)", f"€{hvc_capital_risk/1e6:.2f}M", "Total loss from premium accounts")
            
        st.markdown("---")
        col_hv_chart1, col_hv_chart2 = st.columns(2)
        
        with col_hv_chart1:
            st.markdown("#### Balance vs. Salary Distribution of Premium Customers")
            # Sample for plot performance if huge, but 10k max is fine
            fig_scatter = px.scatter(
                hvc_df,
                x='EstimatedSalary',
                y='Balance',
                color='Status',
                color_discrete_map={'Retained': '#10b981', 'Churned': '#ef4444'},
                opacity=0.6,
                labels={'EstimatedSalary': 'Annual Salary (€)', 'Balance': 'Account Balance (€)'},
                height=400
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color='#f8fafc', gridcolor='#334155'),
                yaxis=dict(color='#f8fafc', gridcolor='#334155'),
                legend=dict(font=dict(color='#f8fafc')),
                font=dict(color='#f8fafc')
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("📈 Scatter analysis displays churn distribution across financial dimensions. Note the clusters of red dots representing premium flight.")
            
        with col_hv_chart2:
            st.markdown("#### High-Value Churn by Product Count")
            hvc_prod = hvc_df.groupby('NumOfProducts')['Exited'].agg(['mean', 'count']).reset_index()
            hvc_prod['Churn Rate (%)'] = (hvc_prod['mean'] * 100).round(2)
            
            fig_hvc_prod = px.bar(
                hvc_prod,
                x='NumOfProducts',
                y='Churn Rate (%)',
                text='Churn Rate (%)',
                height=400
            )
            fig_hvc_prod.update_traces(textposition='outside', marker_color='#fbbf24')
            fig_hvc_prod.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='Number of Products Owned', color='#f8fafc', gridcolor='#334155'),
                yaxis=dict(title='Churn Rate (%)', color='#f8fafc', gridcolor='#334155', range=[0, max(hvc_prod['Churn Rate (%)'].max()*1.2, 10)]),
                font=dict(color='#f8fafc')
            )
            st.plotly_chart(fig_hvc_prod, use_container_width=True)
            st.caption("📦 **Product Saturation Risk**: Premium customers holding 3 or 4 products have near-complete churn rates, reflecting service friction or cross-selling failures.")
# ----------------- TAB 4: DYNAMIC SEGMENTATION -----------------
with tab4:
    st.markdown("### Multi-Dimensional Segmentation Explorer")
    st.markdown("Select segmentation variables to dynamically break down overall churn and calculate the **Risk Indexes**.")
    
    seg_var = st.selectbox("Select Segment Dimension", [
        "AgeSegment", "CreditBand", "TenureGroup", "BalanceSegment", "Geography", "Gender"
    ])
    
    seg_summary = df.groupby(seg_var).agg(
        Total_Customers=('CustomerId', 'count'),
        Churned_Customers=('Exited', 'sum'),
        Average_Balance=('Balance', 'mean'),
        Average_Salary=('EstimatedSalary', 'mean')
    ).reset_index()
    
    seg_summary['Churn Rate (%)'] = ((seg_summary['Churned_Customers'] / seg_summary['Total_Customers']) * 100).round(2)
    seg_summary['Customer Share (%)'] = ((seg_summary['Total_Customers'] / total_customers) * 100).round(1)
    
    # Calculate Risk Index relative to average
    avg_churn = overall_churn_rate * 100
    seg_summary['Risk Index (100 = Avg)'] = ((seg_summary['Churn Rate (%)'] / avg_churn) * 100).round(0) if avg_churn > 0 else 100
    
    # Re-order columns for display
    display_cols = [
        seg_var, 'Total_Customers', 'Customer Share (%)', 'Churned_Customers', 
        'Churn Rate (%)', 'Risk Index (100 = Avg)', 'Average_Balance'
    ]
    
    formatted_df = seg_summary[display_cols].rename(columns={
        seg_var: 'Segment Value',
        'Total_Customers': 'Total Customers',
        'Churned_Customers': 'Churned Volume',
        'Average_Balance': 'Average Balance (€)'
    })
    
    # Highlight highest risk
    st.dataframe(
        formatted_df.style.background_gradient(subset=['Churn Rate (%)', 'Risk Index (100 = Avg)'], cmap='Reds')
        .format({'Average Balance (€)': '€{:,.2f}'}),
        use_container_width=True
    )
    
    st.markdown("""
    💡 **How to interpret Risk Index**:
    - **100**: Churn rate perfectly aligns with the regional average.
    - **>100**: Elevated churn risk. (e.g., 200 represents twice the baseline risk rate).
    - **<100**: Stable, loyal customer group.
    """)
    
    st.markdown("#### Visualizing Segment Contribution vs. Risk")
    fig_bubble = px.scatter(
        seg_summary,
        x='Total_Customers',
        y='Churn Rate (%)',
        size='Average_Balance',
        color=seg_var,
        text=seg_var,
        labels={'Total_Customers': 'Segment Size (Customer Count)', 'Churn Rate (%)': 'Churn Rate (%)'},
        height=400
    )
    fig_bubble.update_traces(textposition='top center')
    fig_bubble.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(color='#f8fafc', gridcolor='#334155'),
        yaxis=dict(color='#f8fafc', gridcolor='#334155'),
        font=dict(color='#f8fafc')
    )
    st.plotly_chart(fig_bubble, use_container_width=True)
    st.caption("🫧 Bubble size represents the average account balance of the segment. Keep an eye on segments in the top-right quadrant (large segment + high churn).")
# ----------------- TAB 5: INTERACTIVE CHURN PREDICTOR -----------------
with tab5:
    st.markdown("### Machine Learning Churn Risk Predictor")
    st.markdown("This module trains an active **Random Forest Classifier** on the database and uses it to evaluate new banking customer risk profiles dynamically.")
    
    # Train the Model
    @st.cache_resource
    def train_churn_model(data_df):
        # Drop columns not useful for prediction
        X = data_df.drop(columns=['CustomerId', 'Surname', 'Exited', 'Status', 'AgeSegment', 'CreditBand', 'TenureGroup', 'BalanceSegment', 'ActivityStatus', 'CreditCardStatus'])
        y = data_df['Exited']
        
        # One-hot encode Geography and Gender
        X_encoded = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
        
        # Split & Standardize
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Random Forest model
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        rf.fit(X_train_scaled, y_train)
        
        # Model Accuracy
        acc = rf.score(X_test_scaled, y_test)
        
        return rf, scaler, X_encoded.columns.tolist(), acc
    try:
        model, scaler, feature_names, accuracy = train_churn_model(df_raw)
        
        # Split into columns for inputs vs results
        col_pred_left, col_pred_right = st.columns([2, 3])
        
        with col_pred_left:
            st.markdown("#### Input Customer Profile")
            
            p_geo = st.selectbox("Geography / Region", ["France", "Germany", "Spain"])
            p_gender = st.selectbox("Gender", ["Male", "Female"])
            p_age = st.slider("Age", 18, 90, 38)
            p_credit = st.slider("Credit Score", 350, 850, 640)
            p_tenure = st.slider("Tenure (Years)", 0, 10, 4)
            p_balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=500000.0, value=75000.0, step=5000.0)
            p_products = st.slider("Num of Products Owned", 1, 4, 1)
            p_card = st.radio("Has Credit Card?", ["Yes", "No"])
            p_active = st.radio("Is Active Member?", ["Yes", "No"])
            
            # Map values
            p_active_val = 1 if p_active == "Yes" else 0
            p_card_val = 1 if p_card == "Yes" else 0
            p_salary = 100000.0 # Standard salary baseline
            
        with col_pred_right:
            st.markdown("#### Risk Prediction Output")
            st.info(f"⚡ **Model Quality**: Random Forest Classifier trained with **{accuracy*100:.2f}% accuracy**.")
            
            # Prepare row for prediction
            input_dict = {
                'CreditScore': p_credit,
                'Age': p_age,
                'Tenure': p_tenure,
                'Balance': p_balance,
                'NumOfProducts': p_products,
                'HasCrCard': p_card_val,
                'IsActiveMember': p_active_val,
                'EstimatedSalary': p_salary,
                'Geography_Germany': 1 if p_geo == "Germany" else 0,
                'Geography_Spain': 1 if p_geo == "Spain" else 0,
                'Gender_Male': 1 if p_gender == "Male" else 0
            }
            
            input_df = pd.DataFrame([input_dict])
            # Align column order with trained model features
            input_df = input_df[feature_names]
            
            # Scaled prediction
            input_scaled = scaler.transform(input_df)
            prob_exit = model.predict_proba(input_scaled)[0][1]
            risk_pct = prob_exit * 100
            
            # Gauge chart for risk
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_pct,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Customer Churn Probability", 'font': {'size': 20, 'color': '#f8fafc'}},
                number = {'suffix': "%", 'font': {'size': 44, 'color': '#ef4444' if risk_pct >= 50 else '#38bdf8'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#ef4444" if risk_pct >= 50 else "#38bdf8"},
                    'bgcolor': "#1e293b",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': '#10b981'},
                        {'range': [30, 60], 'color': '#fbbf24'},
                        {'range': [60, 100], 'color': '#f43f5e'}
                    ],
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                margin=dict(l=30, r=30, t=50, b=30)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk Evaluation Alert
            if risk_pct < 30:
                st.success("🟢 **Risk Tier: LOW**. Customer exhibits normal, loyal parameters. Standard marketing retention campaigns apply.")
            elif 30 <= risk_pct < 60:
                st.warning("🟡 **Risk Tier: MEDIUM**. Customer exhibits early churn markers (e.g. low activity, single product or high-risk region). Recommend proactive engagement.")
            else:
                st.error("🔴 **Risk Tier: HIGH**. High probability of customer departure. Immediate outreach, fee waiver, or preferential pricing is recommended.")
                
            # Print feature importances
            st.markdown("#### Primary Predictor Weights (Feature Importance)")
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance (%)': (importances * 100).round(2)
            }).sort_values('Importance (%)', ascending=False)
            
            fig_imp = px.bar(
                feat_imp_df.head(6),
                x='Importance (%)',
                y='Feature',
                orientation='h',
                height=250,
                color='Importance (%)',
                color_continuous_scale='Blues'
            )
            fig_imp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(color='#f8fafc', gridcolor='#334155'),
                yaxis=dict(color='#f8fafc', gridcolor='#334155'),
                coloraxis_showscale=False,
                font=dict(color='#f8fafc')
            )
            st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        st.error(f"Error training model: {e}")
        st.info("Ensure the dataset has been generated and is stored as 'churn_modelling.csv' in the project directory.")
# ----------------- TAB 6: POLICY RECOMMENDATIONS -----------------
with tab6:
    st.markdown("### European Central Bank Policy Guidance Framework")
    st.write("Based on systematic data modeling, we outline structural retention guidelines for European retail banks to improve financial system stability and minimize capital flight.")
    
    col_rec1, col_rec2 = st.columns(2)
    
    with col_rec1:
        st.markdown("""
        #### 🗺️ 1. Regional Policy: The Germany Risk Index
        - **Findings**: German customer churn is consistently double (21.4%) that of France and Spain.
        - **Policy Recommendation**: 
          - Commercial banks in Germany must investigate fee structures, digital app experience, or localized competitor offerings.
          - The ECB recommends launching a localized retail audit on product fees, which may be driving yield-sensitive customers away.
        
        #### 👥 2. Demographic Target: Mid-Career Assets (Ages 46-60)
        - **Findings**: Customer churn peaks drastically at 46-60 years old. This cohort holds high balances, representing maximum capital flight risk.
        - **Policy Recommendation**:
          - Shift marketing spend from generic youth acquisition to wealth-retaining products.
          - Offer personalized loyalty and retirement advisory services, fee waivers on premium credit cards, and family banking bundles.
        """)
        
    with col_rec2:
        st.markdown("""
        #### 📦 3. Product Saturation: The Cross-Selling Trap
        - **Findings**: Multi-product customers with 3 or 4 products display extremely high churn rates (>54% and >90% respectively).
        - **Policy Recommendation**:
          - Standard banking strategies promote cross-selling. However, our analytics indicate *forced* cross-selling leads to customer dissatisfaction, administrative errors, or high costs.
          - Banks must audit customer satisfaction on multi-product packages and streamline billing/account reconciliation workflows.
        
        #### ⚡ 4. Engagement: Active Member Conversion
        - **Findings**: Inactive members are far more likely to exit the bank.
        - **Policy Recommendation**:
          - Establish automated trigger campaigns for customers showing declining activity (no transactions for 60+ days).
          - Redesign mobile apps with smart alerts, gamified rewards, or targeted savings prompts to drive daily logins.
        """)
        
    st.info("📋 **ECB Stakeholder Note**: For a complete list of statistical proofs, please consult the generated **[research_paper.md](file:///C:/Users/Zahid/.gemini/antigravity/scratch/customer_churn_analytics/research_paper.md)** and the **[executive_summary.md](file:///C:/Users/Zahid/.gemini/antigravity/scratch/customer_churn_analytics/executive_summary.md)** files in the workspace directory.")
