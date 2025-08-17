import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from streamlit_option_menu import option_menu
import statsmodels.api as sm

# --------------------------------#
# 가상 데이터 생성 함수 #
# --------------------------------#

def generate_virtual_customer_data():
    """ABC회사의 가상 고객사 데이터를 생성하는 함수"""
    np.random.seed(42)
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    contract_date = datetime.date(2024, 1, 1)
    
    dates = pd.to_datetime(pd.date_range(start_date, end_date, freq='W'))
    data = []
    
    products = {
        "EXTRA VIRGIN OLIVE OIL_SPAIN": (10.5, 9.5),
        "VIRGIN OLIVE OIL [ITALY]": (9.0, 8.2),
        "PURE OLIVE OIL (1L)": (8.0, 7.0),
        "POMACE OLIVE OIL_GR": (6.5, 5.8),
        "ORGANIC EVOO 750ML": (12.0, 11.0)
    }
    
    for date in dates:
        for product, (price_before, price_after) in products.items():
            price = price_before if date.date() < contract_date else price_after
            unit_price = np.random.normal(price, 0.5)
            volume = np.random.randint(500, 3000)
            
            data.append({
                "Date": date,
                "Raw Importer Name": "ABC회사",
                "Reported Product Name": f"{product}_{np.random.choice(['A', 'B', 'C'])}",
                "Volume": volume,
                "Unit Price": unit_price
            })
            
    return pd.DataFrame(data)

def generate_virtual_market_data():
    """올리브유 시장의 가상 데이터를 생성하는 함수"""
    np.random.seed(0)
    start_date = datetime.date(2023, 1, 1)
    end_date = datetime.date(2024, 12, 31)
    dates = pd.to_datetime(pd.date_range(start_date, end_date, freq='M'))
    
    importers = ["ABC회사", "CJ제일제당", "동원F&B", "오뚜기", "샘표식품", "대상주식회사"]
    exporters = ["ACME SPAIN", "ITALIANO OIL", "GREECE EXPORTS", "TURKEY OLIVES"]
    countries = ["Spain", "Italy", "Greece", "Turkey"]
    
    data = []
    
    base_prices = {
        "ABC회사": 9.2,
        "CJ제일제당": 9.0,
        "동원F&B": 9.5,
        "오뚜기": 9.8,
        "샘표식품": 10.1,
        "대상주식회사": 9.6
    }

    for date in dates:
        for importer in importers:
            num_transactions = np.random.randint(2, 5)
            for _ in range(num_transactions):
                exporter = np.random.choice(exporters)
                country = np.random.choice(countries)
                base_price = base_prices[importer]
                
                # 시간에 따른 가격 변동성 추가
                price_fluctuation = np.sin(date.month * np.pi / 6) * 0.5
                unit_price = np.random.normal(base_price + price_fluctuation, 0.4)
                volume = np.random.randint(2000, 10000)

                data.append({
                    "Date": date,
                    "Raw Importer Name": importer,
                    "Reported Product Name": "OLIVE OIL",
                    "Volume": volume,
                    "Unit Price": unit_price,
                    "Exporter": exporter,
                    "Origin Country": country
                })
                
    return pd.DataFrame(data)

# --------------------------------#
# 데이터 전처리 및 분석 함수 (기존 코드와 동일) #
# --------------------------------#

def preprocess_product_name(name):
    """'REPORTED PRODUCT NAME'을 정제하는 함수"""
    if not isinstance(name, str): return ''
    name = re.sub(r'\[.*?\]', '', name)
    name = name.split('_')[0]
    name = re.sub(r'(\(?\s*\d+\.?\d*\s*(kg|g|l|ml)\s*\)?)', '', name, flags=re.I)
    name = re.sub(r'[^A-Za-z0-9가-힣]', '', name)
    return name.strip()

def get_cluster_name(cluster_labels, preprocessed_names):
    """각 클러스터의 이름을 생성하는 함수"""
    cluster_name_map = {}
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        if label != -1:
            names_in_cluster = preprocessed_names[cluster_labels == label]
            if len(names_in_cluster) > 0:
                most_common_name = Counter(names_in_cluster).most_common(1)[0][0]
                cluster_name_map[label] = most_common_name
            else:
                cluster_name_map[label] = f'Cluster {label}'
    final_cluster_names = {}
    name_counts = Counter(cluster_name_map.values())
    used_names = {}
    for label, name in cluster_name_map.items():
        if name_counts[name] > 1:
            if name not in used_names: used_names[name] = 1
            final_cluster_names[label] = f"{name}_{used_names[name]}"
            used_names[name] += 1
        else:
            final_cluster_names[label] = name
    final_cluster_names[-1] = 'Noise'
    return final_cluster_names

def remove_outliers_iqr(df, column_name):
    """IQR 방식을 사용하여 이상치를 제거하는 함수"""
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial_rows = len(df)
    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    removed_rows = initial_rows - len(df_filtered)
    if removed_rows > 0:
        st.warning(f"분석의 정확도를 위해 시장 데이터의 단가(Unit Price) 이상치 {removed_rows}건을 제거했습니다.")
    return df_filtered

def generate_summary_table_html(df, group_by_col, header_name, value_col='unit_price'):
    """박스플롯에 대한 요약 테이블 HTML을 생성하는 함수"""
    if df.empty:
        return "<p>요약할 데이터가 없습니다.</p>"
    summary_df = df.groupby(group_by_col)[value_col].agg(['max', 'mean', 'min']).reset_index()
    
    html = f"""
    <style>
        .summary-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .summary-table th, .summary-table td {{
            border: 1px solid #e6e6e6;
            padding: 8px;
            text-align: left;
        }}
        .summary-table th {{
            background-color: #f2f2f2;
        }}
    </style>
    <table class="summary-table">
        <thead>
        <tr>
            <th rowspan="2" style="text-align: center; vertical-align: middle;">{header_name}</th>
            <th colspan="3" style="text-align: center;">수입 단가(USD/KG)</th>
        </tr>
        <tr>
            <th style="text-align: center;">최대</th>
            <th style="text-align: center;">평균</th>
            <th style="text-align: center;">최소</th>
        </tr>
        </thead>
        <tbody>
    """

    for index, row in summary_df.iterrows():
        html += f"""
        <tr>
            <td>{row[group_by_col]}</td>
            <td style="text-align: right;">${row['max']:.2f}</td>
            <td style="text-align: right;">${row['mean']:.2f}</td>
            <td style="text-align: right;">${row['min']:.2f}</td>
        </tr>
        """
    
    html += "</tbody></table>"
    return html

def reset_analysis_states():
    """모든 분석 상태를 초기화하는 함수"""
    st.session_state.analysis_done = False
    keys_to_reset = ['customer_name', 'plot_df', 'customer_df', 'contract_date', 
                    'tfidf_matrix', 'savings_df', 'total_savings', 'market_df',
                    'analyzed_product_name']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def reset_market_analysis_states():
    """목표 2 분석 상태만 초기화하는 함수"""
    st.session_state.market_analysis_done = False
    keys_to_reset = ['market_df', 'analyzed_product_name', 'selected_customer', 
                    'market_contract_date', 'top_competitors_list',
                    'all_competitors_ranked']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

# --------------------------#
# 메인 애플리케이션 UI 및 로직 #
# --------------------------#

st.set_page_config(layout="wide")

# --- 세션 상태 초기화 ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'market_analysis_done' not in st.session_state:
    st.session_state.market_analysis_done = False

# --- 사이드바 메뉴 ---
with st.sidebar:
    selected = option_menu(
        menu_title="메뉴",
        options=["고객사 효율 분석", "시장 경쟁력 분석"],
        icons=["person-bounding-box", "graph-up-arrow"],
        menu_icon="cast",
        default_index=0,
    )
    st.markdown("---")
    st.info("기능 확인용 데모 버전입니다. \n\n- 가상의 데이터를 기반으로 동작합니다. \n- 실제 파일 업로드 기능은 제거되었습니다.")
    st.markdown(
        """
        <div style="text-align: center; color: grey; font-size: 0.8rem;">
            © Made by Seungha Lee
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 페이지 1: 고객사 효율 분석
# ==============================================================================
if selected == "고객사 효율 분석":
    st.title('💲 고객사 효율 분석')
    
    if st.session_state.analysis_done:
        st.button("분석 초기화", on_click=reset_analysis_states)
    
    if not st.session_state.analysis_done:
        st.header("⚙️ 기능 확인용 데모 설정")
        st.markdown("`ABC회사`를 고객사로 가정하고, 계약일 전후의 `올리브유` 품목군에 대한 구매 효율성 변화를 분석합니다.")

        if st.button("분석 실행", key="demo1_run"):
            with st.spinner('가상 데이터를 생성하고 분석 중입니다...'):
                df = generate_virtual_customer_data()
                
                rename_dict = {'Date': 'date', 'Reported Product Name': 'product_name', 'Volume': 'volume', 'Unit Price': 'unit_price', 'Raw Importer Name': 'importer_name'}
                df.rename(columns=rename_dict, inplace=True)
                
                df['date'] = pd.to_datetime(df['date'])
                df['year_month'] = df['date'].dt.to_period('M')
                df['year'] = df['date'].dt.year
                df = df.dropna(subset=['importer_name', 'product_name', 'volume', 'unit_price'])
                
                customer_name = "ABC회사"
                contract_date_input = datetime.date(2024, 1, 1)

                customer_df = df[df['importer_name'] == customer_name].copy()
                customer_df['product_preprocessed'] = customer_df['product_name'].apply(preprocess_product_name)
                vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                tfidf_matrix = vectorizer.fit_transform(customer_df['product_preprocessed'])
                dbscan = DBSCAN(eps=0.9, min_samples=3, metric='cosine')
                cluster_labels = dbscan.fit_predict(tfidf_matrix)
                cluster_name_map = get_cluster_name(cluster_labels, customer_df['product_preprocessed'])
                customer_df['cluster'] = cluster_labels
                customer_df['cluster_name'] = customer_df['cluster'].map(cluster_name_map)
                plot_df = customer_df[customer_df['cluster'] != -1].copy()

                contract_date = pd.to_datetime(contract_date_input)
                before_contract_df = plot_df[plot_df['date'] < contract_date]
                after_contract_df = plot_df[plot_df['date'] >= contract_date]
                avg_price_before = before_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_before')
                avg_price_after = after_contract_df.groupby('cluster_name')['unit_price'].mean().rename('avg_price_after')
                volume_after = after_contract_df.groupby('cluster_name')['volume'].sum().rename('volume_after')
                savings_df = pd.concat([avg_price_before, avg_price_after, volume_after], axis=1).dropna()
                savings_df['savings'] = (savings_df['avg_price_before'] - savings_df['avg_price_after']) * savings_df['volume_after']
                savings_df = savings_df.sort_values('savings', ascending=False)
                total_savings = savings_df['savings'].sum()

                st.session_state.customer_name = customer_name
                st.session_state.plot_df = plot_df
                st.session_state.customer_df = customer_df
                st.session_state.contract_date = contract_date
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.savings_df = savings_df
                st.session_state.total_savings = total_savings
                st.session_state.analysis_done = True
                
            st.success(f"'{customer_name}' 고객사 분석 완료!")
            st.rerun()

    if st.session_state.analysis_done:
        with st.expander("1. 계약 전후 예상 절감액 분석", expanded=True):
            st.subheader("총 예상 절감액")
            total_savings = st.session_state.total_savings
            color = "blue" if total_savings >= 0 else "red"
            st.markdown(f"## <span style='color:{color};'>${total_savings:,.2f}</span>", unsafe_allow_html=True)
            st.caption(f"※ 계약일({st.session_state.contract_date.date()}) 이후, 고객사의 자체 구매 단가 변화에 따른 총 예상 절감액입니다.")
            st.subheader("품목군별 상세 절감 내역")
            cols = st.columns(4)
            for i, row in enumerate(st.session_state.savings_df.itertuples()):
                col = cols[i % 4]
                color, arrow, val = ("blue", "▼", row.savings) if row.savings >= 0 else ("red", "▲", -row.savings)
                col.markdown(f"""<div style="border: 1px solid #e6e6e6; border-radius: 0.5rem; padding: 1rem; text-align: center; height: 120px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 1rem;"><strong>{row.Index}</strong><p style="font-size: 1.5rem; font-weight: bold; color: {color}; margin-top: 8px; margin-bottom: 0;">{arrow} ${val:,.0f}</p></div>""", unsafe_allow_html=True)
            
            st.dataframe(st.session_state.savings_df.style.format({
                'avg_price_before': '${:,.2f}',
                'avg_price_after': '${:,.2f}',
                'volume_after': '{:,.0f} KG',
                'savings': '${:,.2f}'
            }))

        with st.expander("2. 수입 품목군 정제 및 군집화 (DBSCAN & PCA)"):
            if st.session_state.get('tfidf_matrix') is not None and st.session_state.tfidf_matrix.shape[0] > 0:
                pca = PCA(n_components=2, random_state=42)
                components = pca.fit_transform(st.session_state.tfidf_matrix.toarray())
                vis_df = pd.DataFrame(components, columns=['x', 'y'])
                vis_df['cluster_name'] = st.session_state.customer_df['cluster_name'].values
                vis_df['product_name'] = st.session_state.customer_df['product_name'].values
                cluster_volume_sorted = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False)
                top_clusters_for_viz = cluster_volume_sorted.head(15).index.tolist()
                vis_df_filtered = vis_df[vis_df['cluster_name'].isin(top_clusters_for_viz)]
                st.info(f"클러스터가 너무 많아, 수입량 기준 상위 {len(top_clusters_for_viz)}개 품목군만 그리드에 시각화합니다.")
                fig1 = px.scatter(vis_df_filtered[vis_df_filtered['cluster_name'] != 'Noise'], x='x', y='y', color='cluster_name', facet_col='cluster_name', facet_col_wrap=5, height=800, 
                                    title=f"<b>[{st.session_state.customer_name}] 품목 유사도 기반 군집화 (상위 품목군 Grid)</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 {len(top_clusters_for_viz)}개 품목군</span>", 
                                    labels={'x': 'PCA Component 1', 'y': 'PCA Component 2'}, hover_data=['product_name'])
                fig1.update_traces(marker=dict(size=8, opacity=0.8))
                st.plotly_chart(fig1, use_container_width=True)
                st.subheader("클러스터 리스트 (수입 중량순)")
                plot_df_sorted = st.session_state.plot_df.copy()
                plot_df_sorted['cluster_name'] = pd.Categorical(plot_df_sorted['cluster_name'], categories=cluster_volume_sorted.index.tolist(), ordered=True)
                st.dataframe(plot_df_sorted[['product_name', 'product_preprocessed', 'cluster_name']].drop_duplicates().sort_values('cluster_name'))

        with st.expander("3. 주요 수입 품목군 분석 (월별 수입량)"):
            plot_df_chart = st.session_state.plot_df.copy()
            plot_df_chart['year_month_str'] = plot_df_chart['year_month'].astype(str)
            cluster_volume = plot_df_chart.groupby(['year_month_str', 'cluster_name'])['volume'].sum().reset_index()
            sorted_clusters = st.session_state.plot_df.groupby('cluster_name')['volume'].sum().sort_values(ascending=False).index.tolist()
            fig2 = px.bar(cluster_volume, x='year_month_str', y='volume', color='cluster_name', 
                            title=f"<b>[{st.session_state.customer_name}] 주요 수입 품목군 월별 수입량(KG)</b>", 
                            labels={'year_month_str': '연-월', 'volume': '수입량(KG)', 'cluster_name': '품목 클러스터'}, 
                            category_orders={'cluster_name': sorted_clusters})
            st.plotly_chart(fig2, use_container_width=True)

# ==============================================================================
# 페이지 2: 시장 경쟁력 분석
# ==============================================================================
if selected == "시장 경쟁력 분석":
    st.title('🏆 시장 경쟁력 상세 분석 (올리브유)')
    
    if st.session_state.get('market_analysis_done', False):
        st.button("분석 초기화", on_click=reset_market_analysis_states)

    if not st.session_state.get('market_analysis_done', False):
        st.header("⚙️ 기능 확인용 데모 설정")
        st.markdown("`올리브유` 품목 시장에서 `ABC회사`의 경쟁력을 국내 주요 식품 회사들과 비교 분석합니다.")
        
        if st.button("시장 경쟁력 분석 실행", key="demo2_run"):
            with st.spinner('가상 시장 데이터를 생성하고 분석 중입니다...'):
                market_df = generate_virtual_market_data()
                customer_name_selection = "ABC회사"
                analyzed_product_name_input = "올리브유"
                contract_date_input = datetime.date(2024, 1, 1)

                rename_dict = {'Date': 'date', 'Reported Product Name': 'product_name', 'Volume': 'volume', 'Unit Price': 'unit_price', 'Origin Country': 'origin_country', 'Raw Importer Name': 'importer_name'}
                market_df.rename(columns=rename_dict, inplace=True)
                market_df['date'] = pd.to_datetime(market_df['date'])
                market_df['year_month'] = market_df['date'].dt.to_period('M')
                market_df['year'] = market_df['date'].dt.year
                market_df['quarter'] = market_df['date'].dt.quarter
                
                required_market_cols = ['importer_name', 'product_name', 'volume', 'unit_price', 'Exporter', 'origin_country']
                market_df = market_df.dropna(subset=required_market_cols)
                market_df = remove_outliers_iqr(market_df, 'unit_price')
                
                lowess_results = sm.nonparametric.lowess(market_df['unit_price'], market_df['volume'], frac=0.5)
                market_df['expected_price'] = np.interp(market_df['volume'], lowess_results[:, 0], lowess_results[:, 1])
                market_df['competitiveness_index'] = market_df['expected_price'] - market_df['unit_price']
                
                all_competitors_ranked = market_df.groupby('importer_name')['competitiveness_index'].mean().sort_values(ascending=False).reset_index()
                
                customer_rank_info = all_competitors_ranked[all_competitors_ranked['importer_name'] == customer_name_selection]
                customer_rank = customer_rank_info.index[0] if not customer_rank_info.empty else len(all_competitors_ranked)
                top_competitors_list = all_competitors_ranked.iloc[:customer_rank]['importer_name'].tolist()
                if customer_name_selection in top_competitors_list:
                    top_competitors_list.remove(customer_name_selection)
                
                st.session_state.market_df = market_df
                st.session_state.analyzed_product_name = analyzed_product_name_input
                st.session_state.selected_customer = customer_name_selection
                st.session_state.market_contract_date = pd.to_datetime(contract_date_input)
                st.session_state.top_competitors_list = top_competitors_list
                st.session_state.all_competitors_ranked = all_competitors_ranked
                st.session_state.market_analysis_done = True
            st.rerun()

    if st.session_state.get('market_analysis_done', False):
        customer_name = st.session_state.selected_customer
        market_df = st.session_state.market_df
        analyzed_product_name = st.session_state.analyzed_product_name
        contract_date = st.session_state.market_contract_date
        top_competitors_list = st.session_state.top_competitors_list
        all_competitors_ranked = st.session_state.all_competitors_ranked
        
        st.subheader(f"'{analyzed_product_name}' 품목 시장 분석 결과 (기준 고객사: {customer_name})")

        with st.expander(f"1. [{analyzed_product_name}] 구매 경쟁력 분석", expanded=True):
            st.markdown("##### Volume 대비 Unit Price 분포 및 시장 추세")
            fig_comp = px.scatter(market_df, x='volume', y='unit_price', trendline="lowess", trendline_color_override="red", hover_data=['importer_name', 'date'], 
                                    title="<b>시장 내 거래 분포 및 평균 가격 추세선</b><br><span style='font-size: 0.8em; color:grey;'>LOWESS 회귀분석 기반</span>",
                                    labels={'volume': '수입량(KG)', 'unit_price': '단가(USD/KG)'})
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("##### 구매 경쟁력 상위 10개사")
            top_10_competitors = all_competitors_ranked.head(10)
            
            def highlight_customer(row):
                color = 'background-color: lightblue' if row.importer_name == customer_name else ''
                return [color] * len(row)
            
            st.dataframe(top_10_competitors.style.apply(highlight_customer, axis=1).format({'competitiveness_index': '{:,.2f}'}))
            
            customer_rank_info = all_competitors_ranked[all_competitors_ranked['importer_name'] == customer_name]
            if not customer_rank_info.empty:
                customer_rank = customer_rank_info.index[0] + 1
                if customer_rank > 10:
                    st.info(f"참고: **{customer_name}**의 구매 경쟁력 순위는 전체 {len(all_competitors_ranked)}개사 중 **{customer_rank}위**입니다.")

        with st.expander(f"2. [{analyzed_product_name}] 단가 추세 및 경쟁 우위 그룹 벤치마킹", expanded=True):
            st.markdown("##### 구매 경쟁력 지수 월별 추이")
            monthly_competitiveness = market_df.groupby(['year_month', 'importer_name'])['competitiveness_index'].mean().unstack()
            
            market_avg_monthly_comp = monthly_competitiveness.mean(axis=1)
            customer_monthly_comp = monthly_competitiveness.get(customer_name)
            
            fig_comp_trend = go.Figure()
            fig_comp_trend.add_trace(go.Scatter(x=market_avg_monthly_comp.index.to_timestamp(), y=market_avg_monthly_comp, mode='lines', name='시장 전체 평균 지수', line=dict(color='blue', width=3)))
            if customer_monthly_comp is not None:
                fig_comp_trend.add_trace(go.Scatter(x=customer_monthly_comp.index.to_timestamp(), y=customer_monthly_comp, mode='lines+markers', name=f'{customer_name} 경쟁력 지수', line=dict(color='red')))
            if top_competitors_list:
                top_competitors_monthly_comp = monthly_competitiveness[top_competitors_list]
                top_competitors_avg_monthly_comp = top_competitors_monthly_comp.mean(axis=1)
                fig_comp_trend.add_trace(go.Scatter(x=top_competitors_avg_monthly_comp.index.to_timestamp(), y=top_competitors_avg_monthly_comp, mode='lines+markers', name='경쟁 우위 그룹 평균 지수', line=dict(color='green', dash='dash')))

            fig_comp_trend.update_layout(title=f'<b>[{analyzed_product_name}] 구매 경쟁력 지수 월별 추이</b>', xaxis_title='연-월', yaxis_title='구매 경쟁력 지수')
            st.plotly_chart(fig_comp_trend, use_container_width=True)
            st.caption("※ 이 그래프는 시장의 기대 단가 대비 실제 구매 단가의 차이(경쟁력 지수)가 시간에 따라 어떻게 변하는지를 보여줍니다.")
            st.markdown("---")

            st.markdown("##### 월별 평균 단가 추세")
            market_avg_price = market_df.groupby('year_month')['unit_price'].mean().rename('market_avg_price')
            customer_market_df = market_df[market_df['importer_name'] == customer_name]
            customer_avg_price = customer_market_df.groupby('year_month')['unit_price'].mean().rename('customer_avg_price')
            
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=market_avg_price.index.to_timestamp(), y=market_avg_price, mode='lines+markers', name='시장 전체 평균 단가', line=dict(width=3)))
            fig4.add_trace(go.Scatter(x=customer_avg_price.index.to_timestamp(), y=customer_avg_price, mode='lines+markers', name=f'{customer_name} 평균 단가', line=dict(color='red')))
            
            if top_competitors_list:
                st.info(f"**벤치마크: 경쟁 우위 그룹 평균**")
                st.caption("※ '경쟁 우위 그룹'은 '구매 경쟁력 분석'의 순위에서 현재 선택된 고객사보다 높은 순위를 기록한 모든 기업들의 평균입니다.")
                top_competitors_df = market_df[market_df['importer_name'].isin(top_competitors_list)]
                top_competitors_avg_price = top_competitors_df.groupby('year_month')['unit_price'].mean().rename('top_competitors_avg_price')
                fig4.add_trace(go.Scatter(x=top_competitors_avg_price.index.to_timestamp(), y=top_competitors_avg_price, mode='lines+markers', name='경쟁 우위 그룹 평균', line=dict(color='green', dash='dash')))
            else:
                st.success(f"**벤치마크 분석:** `{customer_name}`님이 현재 시장에서 가장 우수한 구매 경쟁력을 보이고 있습니다!")

            fig4.update_layout(title=f'<b>[{analyzed_product_name}] 단가 추세</b>', xaxis_title='연-월', yaxis_title='평균 단가(USD/KG)')
            st.plotly_chart(fig4, use_container_width=True)

            st.markdown("##### 전체 기간 평균 단가 비교")
            col1, col2, col3 = st.columns(3)
            col1.metric("시장 전체 평균", f"${market_df['unit_price'].mean():.2f}")
            col2.metric(f"{customer_name} 평균", f"${customer_market_df['unit_price'].mean():.2f}")
            if top_competitors_list:
                col3.metric("경쟁 우위 그룹 평균", f"${top_competitors_df['unit_price'].mean():.2f}")
        
        # 이하 코드는 기존과 동일하게 유지됩니다.
        with st.expander(f"3. [{analyzed_product_name}] 시장 점유율 및 경쟁사 비교", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                years_with_data = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data:
                    selected_year_ms = st.selectbox("시장 점유율 분석 연도 선택", options=years_with_data, key=f"ms_year_{analyzed_product_name}")
                    ms_df = market_df[market_df['year'] == selected_year_ms]
                    ms_data = ms_df.groupby('importer_name')['volume'].sum().sort_values(ascending=False).reset_index()
                    display_data = ms_data.head(5)
                    if customer_name not in display_data['importer_name'].tolist() and not ms_data[ms_data['importer_name']==customer_name].empty:
                        customer_data = ms_data[ms_data['importer_name']==customer_name]
                        display_data = pd.concat([customer_data, display_data.head(4)])
                    others_volume = ms_data[~ms_data['importer_name'].isin(display_data['importer_name'])]['volume'].sum()
                    if others_volume > 0: display_data.loc[len(display_data)] = {'importer_name': '기타', 'volume': others_volume}
                    
                    competitors = [imp for imp in display_data['importer_name'] if imp != customer_name and imp != '기타']
                    blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                    color_map_pie = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_pie[customer_name] = 'red'
                    color_map_pie['기타'] = 'lightgrey'
                    
                    fig5 = px.pie(display_data, values='volume', names='importer_name', color='importer_name',
                                    title=f"<b>[{analyzed_product_name}] {selected_year_ms}년 시장 점유율</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준</span>", 
                                    hole=0.3, color_discrete_map=color_map_pie)
                    fig5.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig5, use_container_width=True)
            with col2:
                years_with_data_price = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_price:
                    selected_year_price = st.selectbox("수입 상위 5개사 단가 비교 연도", options=years_with_data_price, key=f"price_year_{analyzed_product_name}")
                    price_comp_df = market_df[market_df['year'] == selected_year_price]
                    top_importers_by_vol = price_comp_df.groupby('importer_name')['volume'].sum().nlargest(5).index.tolist()
                    if customer_name not in top_importers_by_vol: top_importers_by_vol.append(customer_name)
                    price_comp_data = price_comp_df[price_comp_df['importer_name'].isin(top_importers_by_vol)]
                    avg_price_by_importer = price_comp_data.groupby('importer_name')['unit_price'].mean().sort_values().reset_index()
                    
                    competitors = [imp for imp in avg_price_by_importer['importer_name'] if imp != customer_name]
                    blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                    color_map_bar = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                    color_map_bar[customer_name] = 'red'

                    fig6 = px.bar(avg_price_by_importer, x='importer_name', y='unit_price', title=f"<b>{selected_year_price}년 고객사와 수입 상위 5개사 단가 비교</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 5개사</span>", labels={'importer_name': '수입사', 'unit_price': '평균 단가(USD/KG)'}, color='importer_name', color_discrete_map=color_map_bar)
                    st.plotly_chart(fig6, use_container_width=True)
        
        if 'Exporter' in market_df.columns and 'origin_country' in market_df.columns:
            with st.expander(f"4. [{analyzed_product_name}] 공급망(공급사/원산지) 분석", expanded=True):
                years_with_data_exporter = sorted(market_df['year'].unique(), reverse=True)
                if years_with_data_exporter:
                    selected_year_exporter = st.selectbox("공급망 분석 연도 선택", options=years_with_data_exporter, key=f"exporter_year_{analyzed_product_name}")
                    exporter_analysis_df = market_df[market_df['year'] == selected_year_exporter]
                    
                    top_10_exporters_by_vol = exporter_analysis_df.groupby('Exporter')['volume'].sum().nlargest(10).index
                    exporter_analysis_df_top10 = exporter_analysis_df[exporter_analysis_df['Exporter'].isin(top_10_exporters_by_vol)]

                    st.subheader(f"{selected_year_exporter}년 분기별 공급사 단가 분포")
                    fig9 = px.box(exporter_analysis_df_top10, x='quarter', y='unit_price', color='Exporter', 
                                    title=f"<b>{selected_year_exporter}년 분기별 공급사 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 공급사</span>", 
                                    labels={'quarter': '분기', 'unit_price': '단가(USD/KG)'})
                    st.plotly_chart(fig9, use_container_width=True)
                    with st.expander("상세 데이터 보기"):
                        st.markdown(generate_summary_table_html(exporter_analysis_df_top10, 'Exporter', '공급사'), unsafe_allow_html=True)

                    customer_exporters_in_year = exporter_analysis_df[exporter_analysis_df['importer_name'] == customer_name]['Exporter'].unique()
                    st.info(f"**{customer_name}**가 {selected_year_exporter}년에 거래한 공급사: **{', '.join(customer_exporters_in_year)}**")

                    for exporter in customer_exporters_in_year:
                        st.markdown(f"--- \n #### 공급사 '{exporter}' 비교 분석")
                        single_exporter_df = exporter_analysis_df[exporter_analysis_df['Exporter'] == exporter]
                        
                        top_10_importers_by_vol = single_exporter_df.groupby('importer_name')['volume'].sum().nlargest(10).index
                        single_exporter_df_top10 = single_exporter_df[single_exporter_df['importer_name'].isin(top_10_importers_by_vol)]
                        
                        importers_in_plot = single_exporter_df_top10['importer_name'].unique()
                        competitors = [imp for imp in importers_in_plot if imp != customer_name]
                        blue_shades = px.colors.sequential.Blues_r[::(len(px.colors.sequential.Blues_r)//(len(competitors)+1)) if competitors else 1]
                        color_map_box = {comp: blue_shades[i % len(blue_shades)] for i, comp in enumerate(competitors)}
                        color_map_box[customer_name] = 'red'

                        fig10 = px.box(single_exporter_df_top10, x='importer_name', y='unit_price', 
                                        title=f"<b>'{exporter}' 거래 업체별 단가 분포</b><br><span style='font-size: 0.8em; color:grey;'>수입 중량 기준 상위 10개 수입사</span>", 
                                        labels={'importer_name': '수입사', 'unit_price': '단가(USD/KG)'}, color='importer_name', color_discrete_map=color_map_box)
                        st.plotly_chart(fig10, use_container_width=True)
                        with st.expander("상세 데이터 보기"):
                           st.markdown(generate_summary_table_html(single_exporter_df_top10, 'importer_name', '수입사'), unsafe_allow_html=True)
        else:
            st.warning("'Exporter' 또는 'Origin Country' 컬럼이 없어 공급망 분석을 수행할 수 없습니다.")
