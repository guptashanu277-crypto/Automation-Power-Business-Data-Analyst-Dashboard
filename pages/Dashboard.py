# AI-POWERED BUSINESS INTELLIGENCE DASHBOARD
# Replace the existing Dashboard.py with this file.
# Expected integration from the main app:
# st.session_state['logged_in'] = True
# st.session_state['analytics_df'] = uploaded_dataframe

import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings('ignore', category=FutureWarning)
st.set_page_config(page_title='AI Business Intelligence', page_icon='📊', layout='wide')

# -------------------- ACCESS + DATA --------------------
if not st.session_state.get('logged_in', False):
    st.warning('🔐 Please login first to access the Dashboard.')
    st.stop()

source = st.session_state.get('analytics_df')
if not isinstance(source, pd.DataFrame):
    st.warning('Please upload/select a dataset from the main application first.')
    st.stop()
if source.empty:
    st.warning('The uploaded dataset is empty.')
    st.stop()

df = source.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.dropna(axis=1, how='all')
if df.empty or len(df.columns) == 0:
    st.error('No usable columns found in the dataset.')
    st.stop()

# -------------------- COLUMN UNDERSTANDING --------------------
FINANCIAL = ['sales','sale','revenue','profit','income','amount','price','cost','expense','spend','salary','budget','discount','tax','margin','value','payment','refund','turnover','gmv']
VOLUME = ['quantity','qty','units','unit','orders','order','count','customers','customer','employees','employee','patients','transactions','visits','clicks','leads','stock','inventory']
QUALITY = ['rating','score','quality','satisfaction','percentage','percent','rate','duration','age','experience']
DATE_HINTS = ['date','time','timestamp','month','year','day','created','updated','joined']
CATEGORY_HINTS = ['category','type','class','region','city','state','country','department','product','brand','segment','status','gender','group','industry','location','role','channel','source','campaign']
ID_HINTS = ['id','identifier','uuid','serial number','serial no','phone','mobile','zip code','postal code']

def normalize(x):
    return re.sub(r'\s+', ' ', str(x).strip().lower().replace('_',' ').replace('-',' '))

def has_hint(name, hints):
    return any(h in name for h in hints)

def as_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    cleaned = s.astype(str).str.replace(',', '', regex=False).str.replace('$','',regex=False).str.replace('₹','',regex=False).str.replace('%','',regex=False).str.strip()
    return pd.to_numeric(cleaned, errors='coerce')

def identify_columns(data):
    numeric, categorical, dates, identifiers, text, ignored = [], [], [], [], [], []
    for col in list(data.columns):
        s = data[col]
        nonnull = s.dropna()
        name = normalize(col)
        if nonnull.empty:
            ignored.append(col); continue
        if pd.api.types.is_datetime64_any_dtype(s):
            dates.append(col); continue
        if has_hint(name, DATE_HINTS):
            parsed = pd.to_datetime(s, errors='coerce')
            if parsed.notna().mean() >= .65:
                data[col] = parsed; dates.append(col); continue
        unique_ratio = nonnull.nunique() / max(len(nonnull), 1)
        if has_hint(name, ID_HINTS) and unique_ratio >= .5:
            identifiers.append(col); continue
        parsed_num = as_num(s)
        if parsed_num.notna().mean() >= .85 and parsed_num.nunique(dropna=True) > 1:
            data[col] = parsed_num
            numeric.append(col); continue
        unique_count = nonnull.nunique()
        if pd.api.types.is_bool_dtype(s) or unique_count <= min(100, max(20, int(len(data)*.2))) or unique_ratio < .8:
            categorical.append(col)
        elif unique_ratio >= .8:
            text.append(col)
        else:
            ignored.append(col)
    return numeric, categorical, dates, identifiers, text, ignored

numeric, categorical, dates, identifiers, long_text, ignored = identify_columns(df)


def rank_measures(cols, words):
    def score(col):
        name = normalize(col)
        points = sum(8 for w in words if w in name)
        unique = df[col].nunique(dropna=True)
        if unique >= 5: points += 2
        if unique <= 2: points -= 3
        return points
    return sorted(cols, key=lambda c: (score(c), df[c].nunique(dropna=True)), reverse=True)

def rank_dimensions(cols):
    def score(col):
        name = normalize(col); n = df[col].nunique(dropna=True)
        points = sum(6 for w in CATEGORY_HINTS if w in name)
        if 2 <= n <= 12: points += 8
        elif 13 <= n <= 30: points += 5
        elif n > 100: points -= 8
        return points
    return sorted(cols, key=lambda c: (score(c), -df[c].nunique(dropna=True)), reverse=True)

def group_measure(data, dimension, measure, agg='sum'):
    temp = data[[dimension, measure]].copy()
    temp[measure] = as_num(temp[measure])
    grouped = temp.groupby(dimension, dropna=False)[measure]
    if agg == 'mean': out = grouped.mean()
    elif agg == 'count': out = grouped.count()
    else: out = grouped.sum(min_count=1)
    return out.reset_index().sort_values(measure, ascending=False)

def fmt(v):
    try:
        v = float(v)
        if not np.isfinite(v): return 'N/A'
        if abs(v) >= 1_000_000_000: return f'{v/1_000_000_000:,.2f}B'
        if abs(v) >= 1_000_000: return f'{v/1_000_000:,.2f}M'
        if abs(v) >= 100_000: return f'{v/100_000:,.2f}L'
        return f'{v:,.2f}'
    except Exception:
        return str(v)

# -------------------- FILTERS --------------------
st.title('📊 AI-Powered Business Intelligence Dashboard')
st.caption('Automatic data understanding • KPI discovery • Business performance • Advanced analytics')
filtered = df.copy()
with st.sidebar:
    st.header('🎛️ Dashboard Filters')
    if dates:
        date_col = st.selectbox('Date field', ['No date filter'] + dates)
        if date_col != 'No date filter':
            valid = pd.to_datetime(filtered[date_col], errors='coerce').dropna()
            if not valid.empty:
                low, high = valid.min().date(), valid.max().date()
                chosen = st.date_input('Date range', value=(low, high), min_value=low, max_value=high)
                if isinstance(chosen, tuple) and len(chosen) == 2:
                    d = pd.to_datetime(filtered[date_col], errors='coerce')
                    filtered = filtered[d.dt.date.between(chosen[0], chosen[1])]
    for col in rank_dimensions(categorical)[:4]:
        vals = sorted(filtered[col].dropna().astype(str).unique().tolist())
        selected = st.multiselect(f'Filter {col}', vals, default=[], key=f'filter_{col}')
        if selected:
            filtered = filtered[filtered[col].astype(str).isin(selected)]
    st.divider()
    st.caption(f'{len(filtered):,} of {len(df):,} records')
if filtered.empty:
    st.warning('No records match these filters. Change the filters and try again.')
    st.stop()

active_numeric = [c for c in numeric if as_num(filtered[c]).notna().any()]
active_cat = [c for c in categorical if filtered[c].nunique(dropna=True) > 1]
active_dates = [c for c in dates if pd.to_datetime(filtered[c], errors='coerce').notna().any()]
financial_rank = rank_measures(active_numeric, FINANCIAL)
volume_rank = rank_measures(active_numeric, VOLUME)
quality_rank = rank_measures(active_numeric, QUALITY)
dimension_rank = rank_dimensions(active_cat)

# -------------------- NAVIGATION --------------------
sections = ['🏠 Executive Overview','📈 Business Performance','🧩 Dimension Analysis','🧪 Advanced Analytics','🤖 AI Business Insights','🧹 Data Quality','📋 Data Explorer']
with st.sidebar:
    page = st.radio('Dashboard Sections', sections)

# -------------------- 1. EXECUTIVE OVERVIEW --------------------
if page == '🏠 Executive Overview':
    st.subheader('Executive Overview')
    a,b,c,d = st.columns(4)
    a.metric('Total Records', f'{len(filtered):,}')
    b.metric('Total Fields', f'{len(filtered.columns):,}')
    c.metric('Numeric Measures', f'{len(active_numeric):,}')
    d.metric('Business Dimensions', f'{len(active_cat):,}')
    kpis = []
    for col in financial_rank[:2]: kpis.append(('Total', col, as_num(filtered[col]).sum()))
    for col in volume_rank[:1]:
        if col not in [x[1] for x in kpis]: kpis.append(('Total', col, as_num(filtered[col]).sum()))
    for col in quality_rank[:1]:
        if col not in [x[1] for x in kpis]: kpis.append(('Average', col, as_num(filtered[col]).mean()))
    if kpis:
        st.markdown('#### Automatically detected KPIs')
        cards = st.columns(min(4, len(kpis)))
        for i,(kind,col,val) in enumerate(kpis[:4]): cards[i].metric(f'{kind} {col}', fmt(val))
    st.markdown('#### Dataset health snapshot')
    x,y,z = st.columns(3)
    x.metric('Missing Cells', f'{int(filtered.isna().sum().sum()):,}')
    y.metric('Duplicate Rows', f'{int(filtered.duplicated().sum()):,}')
    z.metric('Date Fields', f'{len(active_dates):,}')
    if active_numeric:
        st.markdown('#### Measure summary')
        st.dataframe(filtered[active_numeric[:10]].describe().T.round(2), use_container_width=True)
    if active_cat and active_numeric:
        dim = dimension_rank[0]
        measure = financial_rank[0] if financial_rank else active_numeric[0]
        chart_data = group_measure(filtered, dim, measure).head(12)
        fig = px.bar(chart_data, x=dim, y=measure, color=dim, title=f'{measure} by {dim}', text_auto='.3s')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# -------------------- 2. BUSINESS PERFORMANCE --------------------
elif page == '📈 Business Performance':
    st.subheader('Business Performance')
    if not active_numeric:
        st.info('This section requires at least one numeric measure.')
    else:
        default_measure = financial_rank[0] if financial_rank else active_numeric[0]
        measure = st.selectbox('Primary measure', active_numeric, index=active_numeric.index(default_measure))
        secondary = st.selectbox('Compare with (optional)', ['None'] + [c for c in active_numeric if c != measure])
        if active_cat:
            dim = st.selectbox('Business dimension', active_cat, index=0)
            left,right = st.columns(2)
            with left:
                gd = group_measure(filtered, dim, measure).head(15)
                fig = px.bar(gd, x=dim, y=measure, color=dim, title=f'{measure} by {dim}', text_auto='.3s')
                fig.update_layout(showlegend=False); st.plotly_chart(fig, use_container_width=True)
            with right:
                if secondary != 'None':
                    compare = filtered.groupby(dim, dropna=False)[[measure,secondary]].sum(numeric_only=True).reset_index()
                    melted = compare.melt(id_vars=dim, var_name='Measure', value_name='Value')
                    fig = px.bar(melted, x=dim, y='Value', color='Measure', barmode='group', title=f'{measure} vs {secondary}')
                else:
                    gd = group_measure(filtered, dim, measure).head(8)
                    fig = px.pie(gd, names=dim, values=measure, hole=.4, title=f'{measure} contribution by {dim}')
                st.plotly_chart(fig, use_container_width=True)
        if active_dates:
            date_field = st.selectbox('Trend date field', active_dates)
            temp = filtered[[date_field,measure]].copy()
            temp[date_field] = pd.to_datetime(temp[date_field], errors='coerce'); temp[measure] = as_num(temp[measure]); temp = temp.dropna()
            if not temp.empty:
                temp['Period'] = temp[date_field].dt.to_period('M').dt.to_timestamp()
                trend = temp.groupby('Period', as_index=False)[measure].sum()
                st.plotly_chart(px.line(trend, x='Period', y=measure, markers=True, title=f'{measure} monthly trend'), use_container_width=True)
        if len(active_numeric) >= 2:
            st.markdown('#### Measure relationship')
            xx = st.selectbox('X-axis', active_numeric, key='scatter_x')
            yy = st.selectbox('Y-axis', [c for c in active_numeric if c != xx], key='scatter_y')
            points = filtered[[xx,yy]].apply(pd.to_numeric, errors='coerce').dropna()
            if not points.empty: st.plotly_chart(px.scatter(points, x=xx, y=yy, title=f'{yy} vs {xx}', opacity=.7), use_container_width=True)

# -------------------- 3. DIMENSION ANALYSIS --------------------
elif page == '🧩 Dimension Analysis':
    st.subheader('Dimension Analysis')
    if not active_cat:
        st.info('No categorical dimensions detected.')
    else:
        dim = st.selectbox('Choose dimension', active_cat)
        counts = filtered[dim].fillna('Missing').astype(str).value_counts().head(20).rename_axis(dim).reset_index(name='Records')
        left,right = st.columns(2)
        with left: st.plotly_chart(px.bar(counts, x=dim, y='Records', color=dim, title=f'Record count by {dim}'), use_container_width=True)
        with right: st.plotly_chart(px.pie(counts.head(10), names=dim, values='Records', hole=.4, title=f'{dim} distribution'), use_container_width=True)
        if active_numeric:
            measure = st.selectbox('Measure', active_numeric)
            agg_label = st.selectbox('Aggregation', ['Sum','Average','Count'])
            agg = {'Sum':'sum','Average':'mean','Count':'count'}[agg_label]
            gd = group_measure(filtered, dim, measure, agg).head(20)
            st.plotly_chart(px.bar(gd, x=dim, y=measure, color=dim, title=f'{agg_label} of {measure} by {dim}'), use_container_width=True)
        if len(active_cat) >= 2:
            other = st.selectbox('Compare with another dimension', [c for c in active_cat if c != dim])
            cross = pd.crosstab(filtered[dim].fillna('Missing').astype(str), filtered[other].fillna('Missing').astype(str)).iloc[:15,:15]
            st.plotly_chart(px.imshow(cross, text_auto=True, aspect='auto', title=f'{dim} vs {other}'), use_container_width=True)

# -------------------- 4. ADVANCED ANALYTICS --------------------
elif page == '🧪 Advanced Analytics':
    st.subheader('Advanced Analytics')
    if len(active_numeric) >= 2:
        corr = filtered[active_numeric].apply(pd.to_numeric, errors='coerce').corr(numeric_only=True)
        st.plotly_chart(px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Pearson Correlation Matrix'), use_container_width=True)
        st.caption('Correlation indicates association, not causation.')
    else: st.info('At least two numeric measures are needed for correlation analysis.')
    if active_numeric:
        col = st.selectbox('Analyze distribution', active_numeric)
        vals = as_num(filtered[col]).dropna()
        if not vals.empty:
            a,b,c,d = st.columns(4); a.metric('Mean',fmt(vals.mean())); b.metric('Median',fmt(vals.median())); c.metric('Std. Deviation',fmt(vals.std())); d.metric('Unique Values',f'{vals.nunique():,}')
            st.plotly_chart(px.histogram(x=vals, nbins=30, marginal='box', labels={'x':col}, title=f'Distribution of {col}'), use_container_width=True)
            q1,q3 = vals.quantile(.25),vals.quantile(.75); iqr=q3-q1; out=vals[(vals<q1-1.5*iqr)|(vals>q3+1.5*iqr)]
            st.metric('Potential outliers (IQR method)',f'{len(out):,}')
    if active_dates and active_numeric:
        date_field = st.selectbox('Date field for trend', active_dates)
        value_field = st.selectbox('Measure for trend', active_numeric)
        tmp=filtered[[date_field,value_field]].copy(); tmp[date_field]=pd.to_datetime(tmp[date_field],errors='coerce'); tmp[value_field]=as_num(tmp[value_field]); tmp=tmp.dropna()
        if not tmp.empty:
            tmp['Month']=tmp[date_field].dt.to_period('M').dt.to_timestamp(); monthly=tmp.groupby('Month',as_index=False)[value_field].sum()
            st.plotly_chart(px.line(monthly,x='Month',y=value_field,markers=True,title=f'Monthly {value_field} trend'),use_container_width=True)

# -------------------- 5. DATA-GROUNDED BUSINESS INSIGHTS --------------------
elif page == '🤖 AI Business Insights':
    st.subheader('Business Insights')
    st.caption('Calculated observations based on the uploaded data. These do not claim causes that the dataset cannot prove.')
    insights=[]
    for col in (financial_rank[:3] + [c for c in volume_rank[:2] if c not in financial_rank[:3]]):
        v=as_num(filtered[col]).dropna()
        if not v.empty: insights.append(f'**{col}:** total **{fmt(v.sum())}**, average **{fmt(v.mean())}**, median **{fmt(v.median())}**.')
    for col in dimension_rank[:3]:
        counts=filtered[col].fillna('Missing').astype(str).value_counts()
        if not counts.empty: insights.append(f'**{col}:** most frequent value is **{counts.index[0]}** ({counts.iloc[0]:,} records, {counts.iloc[0]/counts.sum()*100:.1f}%).')
    for col in active_dates[:1]:
        v=pd.to_datetime(filtered[col],errors='coerce').dropna()
        if not v.empty: insights.append(f'**{col}:** data spans **{v.min().date()}** to **{v.max().date()}**.')
    missing_pct=filtered.isna().sum().sum()/max(filtered.size,1)*100
    insights.append(f'**Data completeness:** {100-missing_pct:.1f}% of cells are populated.')
    for i,item in enumerate(insights,1): st.markdown(f'{i}. {item}')
    st.markdown('#### Suggested checks')
    if filtered.isna().sum().sum(): st.write('• Review missing values before making business decisions.')
    if filtered.duplicated().sum(): st.write('• Check duplicate rows to determine whether they are repeated transactions or valid records.')
    if len(active_numeric)>=2: st.write('• Investigate strong correlations in business context; correlation alone does not establish causation.')
    if active_cat and active_numeric: st.write('• Compare high- and low-performing categories using relevant measures.')
    if active_dates: st.write('• Review trends over time for sustained changes or seasonality.')

# -------------------- 6. DATA QUALITY --------------------
elif page == '🧹 Data Quality':
    st.subheader('Data Quality')
    total=filtered.size; missing=int(filtered.isna().sum().sum()); dup=int(filtered.duplicated().sum()); completeness=(1-missing/total)*100 if total else 0
    a,b,c,d=st.columns(4); a.metric('Completeness',f'{completeness:.1f}%'); b.metric('Missing Cells',f'{missing:,}'); c.metric('Duplicate Rows',f'{dup:,}'); d.metric('Rows',f'{len(filtered):,}')
    report=pd.DataFrame({'Column':filtered.columns,'Data Type':[str(filtered[x].dtype) for x in filtered.columns],'Missing':[int(filtered[x].isna().sum()) for x in filtered.columns],'Missing %':[round(filtered[x].isna().mean()*100,2) for x in filtered.columns],'Unique Values':[int(filtered[x].nunique(dropna=True)) for x in filtered.columns]})
    st.dataframe(report.sort_values('Missing %',ascending=False),use_container_width=True,hide_index=True)
    if dup: st.warning(f'{dup:,} duplicate rows found. This dashboard reports them but does not delete them.')
    if missing: st.warning('Missing values found. Review the column-level report.')
    else: st.success('No missing cells found in the filtered dataset.')

# -------------------- 7. DATA EXPLORER --------------------
elif page == '📋 Data Explorer':
    st.subheader('Data Explorer')
    st.dataframe(filtered.head(500),use_container_width=True,hide_index=True)
    a,b,c=st.columns(3)
    with a: st.markdown('**Numeric measures**'); st.write(numeric or 'None')
    with b: st.markdown('**Categorical dimensions**'); st.write(categorical or 'None')
    with c: st.markdown('**Date fields**'); st.write(dates or 'None')
    with st.expander('Other detected fields'):
        st.write('Identifiers:',identifiers or 'None'); st.write('Long text:',long_text or 'None'); st.write('Ignored/empty:',ignored or 'None')

st.divider()
st.caption(f'AI Business Intelligence Dashboard • {len(filtered):,} filtered rows • {len(active_numeric)} measures • {len(active_cat)} dimensions • {len(active_dates)} date fields')
