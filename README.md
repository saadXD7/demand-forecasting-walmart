# 🛒 Walmart Demand Forecasting System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Prophet](https://img.shields.io/badge/Meta-Prophet-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline designed to forecast weekly retail demand. Using historical sales data, the system trains Meta's Prophet algorithm to identify complex seasonal trends (like major US holidays) and predict future sales up to 52 weeks in advance. 

The project culminates in an interactive web application built with Streamlit, allowing business stakeholders to dynamically explore the AI's forecasts.

🔗 **[Click Here to View the Live Application](Insert Your Streamlit Link Here)**

---

## 🏗️ Technical Architecture

* **Data Engineering:** Automated scripts merge fragmented store and department datasets, handling missing values and aggregating total company sales using `pandas`.
* **Machine Learning Model:** Utilizes `prophet` to capture weekly, monthly, and yearly seasonality. The model is specifically configured to anticipate the impact of major US holidays (e.g., Thanksgiving, Christmas) on retail demand.
* **Frontend Dashboard:** A responsive, interactive UI built with `streamlit` and `plotly`, enabling users to filter forecast horizons dynamically via a sidebar slider.
* **Deployment & Version Control:** Managed locally via `git` and hosted on GitHub, with CI/CD integration directly to Streamlit Community Cloud.

---

## 🚀 How to Run Locally

If you want to clone this repository and run the model on your own machine:

**1. Clone the repository:**
```bash
git clone [Insert Your GitHub Repo Link Here]
cd walmart_demand_forecasting