# E-Commerce Product Engagement & Conversion Optimization Dashboard

Dashboard Streamlit ini terdiri dari 4 halaman:

1. **Executive Overview**
   - KPI cards
   - Funnel chart: View → Click → Add to Cart → Purchase
   - Monthly trend: views, clicks, purchases, revenue
   - Summary insight box

2. **Product Performance**
   - Top 10 products by CTR
   - Top 10 products by revenue
   - Top 10 products by purchases
   - Scatter plot: views vs CTR
   - Scatter plot: clicks vs purchase rate

3. **Product Optimization Opportunity**
   - High View, Low CTR
   - High Click, Low Purchase
   - Recommendation table

4. **Category & Brand Performance**
   - CTR by category
   - Revenue by category
   - Purchase rate by category
   - Top brands by revenue
   - Top brands by CTR

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload this project folder to GitHub, then deploy `app.py` using Streamlit Community Cloud.
